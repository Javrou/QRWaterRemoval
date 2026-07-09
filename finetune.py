import time
import torch.optim as optim
from torch.amp import autocast, GradScaler

import zxingcpp

from model.restormer import Restormer
from tool.finetune_data_loader import *
from loss import *
from tool.train_logger import *
from tool.checkpoints import *


def finetune(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Torch:", torch.__version__)

    # ======================
    # Model
    # ======================
    model = Restormer(
        inp_channels=3,
        out_channels=3,
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

    # ======================
    # 加载预训练 v1 模型
    # ======================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )
    num_epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    epoch, step, metrics = load_checkpoint(
        checkpoint_path,
        model,
        optimizer=None,
        scheduler=None,
        device=device
    )
    print("Loaded:", metrics)

    scaler = GradScaler("cuda")

    def zxing_rate(batch_tensor):
        success = 0
        total = batch_tensor.shape[0]
        imgs = batch_tensor.detach().cpu()
        for i in range(total):
            img = imgs[i]
            img = img.permute(1, 2, 0).numpy()
            img = (img * 255) \
                .clip(0, 255) \
                .astype("uint8")
            results = zxingcpp.read_barcodes(img)
            if len(results) > 0:
                success += 1

        return success / total

    # ======================
    # PSNR
    # ======================
    def calculate_psnr(pred, gt):
        mse = F.mse_loss(pred, gt)

        if mse == 0:
            return 100

        return (
                10 *
                torch.log10(
                    1.0 / mse
                )
        ).item()

    # ======================
    # Validation
    # ======================
    def validate(model):
        model.eval()

        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        total_success = 0
        total_num = 0

        with torch.no_grad():
            for inp, tgt in val_loader:
                inp = inp.to(device)
                tgt = tgt.to(device)

                pred = model(inp)
                pred = pred.clamp(0, 1)

                # loss
                loss = compute_loss(pred, tgt, mode="finetune")
                total_loss += loss.item()

                # PSNR
                total_psnr += calculate_psnr(
                    pred,
                    tgt
                )

                # SSIM
                ssim_value = (
                        1 - ssim_loss(pred, tgt)
                )
                total_ssim += ssim_value.item()

                # ZXing
                sr = zxing_rate(pred)

                total_success += (
                        sr * inp.size(0)
                )

                total_num += inp.size(0)

        avg_loss = total_loss / len(val_loader)
        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        zxing = total_success / total_num

        return (
            avg_loss,
            zxing,
            avg_psnr,
            avg_ssim
        )

    # ======================
    # Training
    # ======================
    step_logger = StepLogger("logs/finetune/train_step.csv")
    epoch_logger = EpochLogger("logs/finetune/epoch_metrics.csv")
    global_step = 0
    best_metrics = {
        "loss": float("inf"),
        "zxing": 0.0,
        "psnr": 0.0,
        "ssim": 0.0
    }
    patience = 5
    min_epochs = 5

    min_delta_loss = 0.001
    min_delta_zxing = 0.001

    early_stop_counter = 0

    for epoch in range(num_epochs):
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

                loss = compute_loss(
                    pred,
                    tgt
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1e9
            ).item()
            scaler.step(
                optimizer
            )
            scaler.update()
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
                    grad_norm=grad_norm
                )
            if (i + 1) % 60 == 0:
                with torch.no_grad():
                    pred_vis = pred.clamp(0, 1)
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

        train_loss = (
                running_loss /
                len(train_loader)
        )
        val_loss, val_zxing, val_psnr, val_ssim = validate(model)

        print("\n======================")
        print(f"Epoch {epoch} finished")
        print("Time:", time.time() - t_epoch)
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Validation ZXing Rate: "f"{val_zxing:.4f}")
        print(f"Validation PSNR: "f"{val_psnr:.4f}")
        print(f"Validation SSIM: "f"{val_ssim:.4f}")
        print("======================\n")

        loss_improved = (
                                best_metrics["loss"] - val_loss
                        ) > min_delta_loss

        zxing_improved = (
                                 val_zxing - best_metrics["zxing"]
                         ) > min_delta_zxing

        # ======================
        # checkpoint
        # ======================
        is_best_loss = False
        is_best_zxing = False
        if val_loss < best_metrics["loss"]:
            best_metrics["loss"] = val_loss
            is_best_loss = True

        if val_zxing > best_metrics["zxing"]:
            best_metrics["zxing"] = val_zxing
            best_metrics["psnr"] = val_psnr
            best_metrics["ssim"] = val_ssim
            is_best_zxing = True
        metrics = {
            "loss": val_loss,
            "zxing": val_zxing,
            "psnr": val_psnr,
            "ssim": val_ssim,
            "best_loss": best_metrics["loss"],
            "best_zxing": best_metrics["zxing"],
            "best_psnr": best_metrics["psnr"],
            "best_ssim": best_metrics["ssim"]
        }
        save_checkpoint(
            path="checkpoints/real_latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            best_metrics=metrics
        )
        if is_best_loss:
            save_checkpoint(
                path="checkpoints/real_best_loss.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
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
                optimizer=optimizer,
                scheduler=scheduler,
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
            val_loss=val_loss,
            lr=optimizer.param_groups[0]["lr"],
            zxing=val_zxing,
            psnr=val_psnr,
            ssim=val_ssim,
            best_loss=best_metrics["loss"],
            best_zxing=best_metrics["zxing"],
            best_psnr=best_metrics["psnr"],
            best_ssim=best_metrics["ssim"]
        )

        # ======================
        # Early Stopping
        # ======================
        if loss_improved or zxing_improved:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        print(
            f"EarlyStopping: "
            f"{early_stop_counter}/{patience}"
        )

        if (
                epoch + 1 >= min_epochs
                and
                early_stop_counter >= patience
        ):
            print("\n======================")
            print("Early Stopping")
            print("======================")
            print(
                f"Best ZXing : "
                f"{best_metrics['zxing']:.4f}"
            )
            print(
                f"Best Loss : "
                f"{best_metrics['loss']:.6f}"
            )
            break

    print("Training finished")


if __name__ == "__main__":
    finetune(
        "checkpoints/best_zxing.pth"
    )
