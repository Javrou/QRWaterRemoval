import time
import random
import numpy as np
import torch.optim as optim
from torch.amp import autocast, GradScaler

from train.trainer import Trainer
from model.restormer import Restormer
from tool.pretrain_data_loader import *
from tool.train_logger import *
from tool.checkpoints import *
from tool.validator import *


def pretrain():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Torch:", torch.__version__)

    # 轻量restormer
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

    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    num_epochs = 30
    # 断点续训
    resume = True
    resume_path = "../checkpoints/latest.pth"
    step_logger = StepLogger("../logs/pretrain/train_step.csv")
    epoch_logger = EpochLogger("../logs/pretrain/epoch_metrics.csv")
    patience = 6  # 连续6个epoch无提升
    min_delta_loss = 1e-3  # loss至少下降0.001
    min_delta_zxing = 0.002  # ZXing至少提升0.2%
    min_epochs = 15

    early_stop_zxing = 0.90
    early_stop_counter = 0

    start_epoch = 0
    global_step = 0

    # optimizer with AMP
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    scaler = GradScaler("cuda")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        loss_fn=compute_loss,
        device=device,
        step_logger=step_logger,
        ema=None,
        grad_clip=1e9,
        print_freq=50,
        zxing_freq=100
    )

    best_metrics = {
        "loss": float("inf"),
        "zxing": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
    }

    if resume:
        start_epoch, global_step, ckpt_metrics = load_checkpoint(
            path=resume_path,
            model=model,
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
        print("Resume Training")
        print("Epoch:", start_epoch)
        print("Step :", global_step)
        print("Best ZXing:", best_metrics["zxing"])
        print("=" * 40)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        t_epoch = time.time()

        train_metrics = trainer.train_one_epoch(
            train_loader=train_loader,
            epoch=epoch,
            global_step=global_step,
            mode="pretrain",
            zxing_fn=zxing_rate()
        )
        avg_loss = train_metrics["loss"]
        global_step = train_metrics["global_step"]
        print()
        print("Epoch finished")
        print("Epoch time:", train_metrics["time"])
        print("Avg Loss:", avg_loss)

        # ======================
        # Epoch summary
        # ======================
        avg_loss = running_loss / len(train_loader)
        print("\nEpoch finished")
        print("Epoch time:", time.time() - t_epoch)
        print("Avg Loss:", avg_loss)

        scheduler.step()
        print("Current LR:", optimizer.param_groups[0]['lr'])

        val_metrics = validate(model, val_loader, device, mode="pretrain")
        print("Validation Loss :", val_metrics["loss"])
        print("Validation ZXing:", val_metrics["zxing"])
        print("Validation PSNR :", val_metrics["psnr"])
        print("Validation SSIM :", val_metrics["ssim"])
        print("Validation Binary Accuracy:", val_metrics["binary_acc"])
        # ======================
        # checkpoint
        # ======================
        is_best_loss = False
        is_best_zxing = False
        # ---------- Loss ----------
        loss_improved = (
            best_metrics["loss"] - val_metrics["loss"]
        ) > min_delta_loss
        if loss_improved:
            best_metrics["loss"] = val_metrics["loss"]
            is_best_loss = True
        # ---------- ZXing ----------
        zxing_improved = (
            val_metrics["zxing"] - best_metrics["zxing"]
        ) > min_delta_zxing
        if zxing_improved:
            best_metrics["zxing"] = val_metrics["zxing"]
            best_metrics["psnr"] = val_metrics["psnr"]
            best_metrics["ssim"] = val_metrics["ssim"]
            best_metrics["binary_acc"] = val_metrics["binary_acc"]
            is_best_zxing = True
        # ---------- latest metrics ----------
        latest_metrics = {
            "loss": val_metrics["loss"],
            "zxing": val_metrics["zxing"],
            "psnr": val_metrics["psnr"],
            "ssim": val_metrics["ssim"],
            "binary_acc": val_metrics["binary_acc"],
            "best_loss": best_metrics["loss"],
            "best_zxing": best_metrics["zxing"],
            "best_psnr": best_metrics["psnr"],
            "best_ssim": best_metrics["ssim"]
        }
        # ---------- latest ----------
        save_checkpoint(
            path="../checkpoints/latest.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            step=global_step,
            best_metrics=latest_metrics
        )
        # ---------- best loss ----------
        if is_best_loss:
            save_checkpoint(
                path="../checkpoints/best_loss.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                best_metrics=latest_metrics
            )
            print(f">>> Best Loss Updated : {best_metrics['loss']:.6f}")
        # ---------- best zxing ----------
        if is_best_zxing:
            save_checkpoint(
                path="../checkpoints/best_zxing.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=global_step,
                best_metrics=latest_metrics
            )
            print(f">>> Best ZXing Updated: {best_metrics['zxing']:.4f}")
        epoch_logger.log(
            epoch=epoch,
            train_loss=avg_loss,
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
    print("Pretrain finished")
    return "checkpoints/best_zxing.pth"


if __name__ == "__main__":
    pretrain()
