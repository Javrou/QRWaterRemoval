import time
import torch.optim as optim
from torch.amp import autocast, GradScaler

from model.restormer import Restormer
from tool.finetune_data_loader import *
from tool.train_logger import *
from tool.checkpoints import *
from tool.validator import *
from tool.ema import *
from tool.visualizer import *


def finetune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Torch:", torch.__version__)

    # ======================
    # Model
    # ======================
    model = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=24,
        num_blocks=[2, 2, 2, 3],
        num_refinement_blocks=1,
        heads=[1, 2, 2, 4],
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(device)
    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )
    pretrained_path = "../checkpoints/best_zxing.pth"
    if pretrained_path is not None:
        _, _, metrics = load_checkpoint(
            path=pretrained_path,
            model=model,
            device=device
        )

        print("=" * 40)
        print("Load Pretrained")
        print(pretrained_path)
        print("Before Finetune:", metrics)
        print("=" * 40)
    ema = ModelEMA(model, decay=0.9995)

    num_epochs = 30
    resume = False
    resume_path = "checkpoints/real_latest.pth"

    optimizer = optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        threshold=1e-3,
        min_lr=1e-6
    )
    scaler = GradScaler("cuda")

    step_logger = StepLogger("../logs/finetune/train_step.csv")
    epoch_logger = EpochLogger("../logs/finetune/epoch_metrics.csv")
    best_metrics = {
        "loss": float("inf"),
        "zxing": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }
    patience = 5
    min_epochs = 5
    min_delta_loss = 0.001
    min_delta_zxing = 0.001
    early_stop_zxing = 0.96

    early_stop_counter = 0

    start_epoch = 0
    global_step = 0
    if resume:
        start_epoch, global_step, ckpt_metrics = load_checkpoint(
            path=resume_path,
            model=model,
            ema=ema.ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device
        )
        start_epoch += 1
        if len(ckpt_metrics):
            best_metrics["loss"] = ckpt_metrics["best_loss"]
            best_metrics["zxing"] = ckpt_metrics["best_zxing"]
            best_metrics["psnr"] = ckpt_metrics["best_psnr"]
            best_metrics["ssim"] = ckpt_metrics["best_ssim"]
        print("=" * 40)
        print("Resume Fine-tune")
        print("Epoch :", start_epoch)
        print("Step  :", global_step)
        print("Best ZXing :", best_metrics["zxing"])
        print("=" * 40)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0
        t_epoch = time.time()

        for i, (inp, tgt) in enumerate(train_loader):
            t0 = time.time()
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda"):
                pred = model(inp)
                pred_loss = pred.clamp(0,1)
                loss = compute_loss(pred_loss, tgt, mode="finetune")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                2
            ).item()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item()
            global_step += 1

            if (i + 1) % 10 == 0:
                print(
                    f"[E{epoch} "
                    f"{i + 1}/{len(train_loader)}] "
                    f"loss={loss.item():.4f} "
                    f"time={time.time() - t0:.3f}s "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"grad={grad_norm:.2f}"
                )
                step_logger.log(
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=grad_norm,
                    zxing=None
                )
            if (i + 1) % 100 == 0:
                with torch.no_grad():
                    pred_vis = pred.detach().clamp(0, 1)
                    sr = zxing_rate(pred_vis)
                print(f"[ZXing @ step {i + 1}] {sr:.4f}")
                step_logger.log(
                    epoch=epoch,
                    step=global_step,
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=grad_norm,
                    zxing=sr
                )

        # ======================
        # Epoch validation
        # ======================
        train_loss = (running_loss / len(train_loader))
        val_metrics = validate(ema.ema, val_loader, device, mode="finetune")

        print("\n======================")
        print(f"Epoch {epoch} finished")
        print("Time:", time.time() - t_epoch)
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_metrics['loss']:.6f}")
        print(f"Validation ZXing Rate: {val_metrics['zxing']:.4f}")
        print(f"Validation PSNR: {val_metrics['psnr']:.4f}")
        print(f"Validation SSIM: {val_metrics['ssim']:.4f}")
        print(f"Validation Binary Accuracy: {val_metrics['binary_acc']:.4f}")
        print("======================\n")
        scheduler.step(val_metrics["zxing"])
        save_visual_results(
            model=ema.ema,
            loader=val_loader,
            device=device,
            epoch=epoch,
            save_dir="visual/finetune",
            indices=(0, 10, 20, 50)
        )
        # ======================
        # checkpoint
        # ======================
        is_best_loss = False
        is_best_zxing = False
        loss_improved = (
            best_metrics["loss"] - val_metrics["loss"]
        ) > min_delta_loss
        zxing_improved = (
            val_metrics["zxing"] - best_metrics["zxing"]
        ) > min_delta_zxing

        if val_metrics["loss"] < best_metrics["loss"]:
            best_metrics["loss"] = val_metrics["loss"]
            is_best_loss = True
        if val_metrics["zxing"] > best_metrics["zxing"]:
            best_metrics["zxing"] = val_metrics["zxing"]
            best_metrics["psnr"] = val_metrics["psnr"]
            best_metrics["ssim"] = val_metrics["ssim"]
            best_metrics["binary_acc"] = val_metrics["binary_acc"]
            is_best_zxing = True

        metrics = {
            "loss": val_metrics["loss"],
            "zxing": val_metrics["zxing"],
            "psnr": val_metrics["psnr"],
            "ssim": val_metrics["ssim"],
            "binary_acc": val_metrics["binary_acc"],
            "best_loss": best_metrics["loss"],
            "best_zxing": best_metrics["zxing"],
            "best_psnr": best_metrics["psnr"],
            "best_ssim": best_metrics["ssim"],
        }
        save_checkpoint(
            path="checkpoints/real_latest.pth",
            model=model,
            ema=ema.ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            best_metrics=metrics
        )
        if is_best_loss:
            save_checkpoint(
                path="checkpoints/real_best_loss.pth",
                model=model,
                ema=ema.ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                best_metrics=metrics
            )
            print(
                f">>> Best Loss Updated: "
                f"{best_metrics['loss']:.6f}"
            )
        if is_best_zxing:
            save_checkpoint(
                path="checkpoints/real_best_zxing.pth",
                model=model,
                ema=ema.ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                best_metrics=metrics
            )
            print(
                f">>> Best ZXing Updated: "
                f"{best_metrics['zxing']:.4f}"
            )
        epoch_logger.log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            lr=optimizer.param_groups[0]["lr"],
            zxing=val_metrics["zxing"],
            psnr=val_metrics["psnr"],
            ssim=val_metrics["ssim"],
            binary_acc=val_metrics["binary_acc"],
            best_loss=best_metrics["loss"],
            best_zxing=best_metrics["zxing"],
            best_psnr=best_metrics["psnr"],
            best_ssim=best_metrics["ssim"]
        )
        # ======================
        # Early Stopping
        # ======================
        if best_metrics["zxing"] < early_stop_zxing:
            early_stop_counter = 0
            print(
                f"EarlyStopping Disabled "
                f"(Best ZXing={best_metrics['zxing']:.4f})"
            )
        else:
            if loss_improved or zxing_improved:
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            print(f"EarlyStopping: {early_stop_counter}/{patience}")
            if epoch + 1 >= min_epochs and early_stop_counter >= patience:
                print("\n==============================")
                print("Early Stopping Triggered")
                print("==============================")
                print(f"Best ZXing : {best_metrics['zxing']:.4f}")
                print(f"Best Loss  : {best_metrics['loss']:.6f}")
                break
    print("train finished")


if __name__ == "__main__":
    finetune()
