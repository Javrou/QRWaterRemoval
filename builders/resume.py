import os
import torch

from engine.checkpoint import load_checkpoint


def resume_training(cfg, model, optimizer, scheduler, scaler, ema, device):

    start_epoch = 0
    global_step = 0

    best_metrics = {
        "loss": float("inf"),
        "zxing": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "binary_acc": 0.0
    }

    if not cfg.resume:
        return start_epoch, global_step, best_metrics

    # 文件不存在时优雅跳过（首次运行）
    if not os.path.exists(cfg.resume_path):
        return start_epoch, global_step, best_metrics

    start_epoch, global_step, metrics = load_checkpoint(
        path=cfg.resume_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema.ema if ema else None,
        device=device
    )

    start_epoch += 1

    if len(metrics):
        best_metrics["loss"] = metrics["best_loss"]
        best_metrics["zxing"] = metrics["best_zxing"]
        best_metrics["psnr"] = metrics["best_psnr"]
        best_metrics["ssim"] = metrics["best_ssim"]
        best_metrics["binary_acc"] = metrics.get("best_binary_acc", 0)

    print("=" * 40)
    print("Resume Training")
    print("Epoch :", start_epoch)
    print("Step  :", global_step)
    print("Best ZXing :", best_metrics["zxing"])
    print("=" * 40)

    return start_epoch, global_step, best_metrics


def load_model(path, model, ema=None, device="cuda", label="Model"):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(
        ckpt["model"],
        strict=True
    )
    if ema is not None:
        ema.ema.load_state_dict(
            ckpt["model"],
            strict=True
        )

    print("=" * 40)
    print("Loaded " + label)
    print(path)
    print("=" * 40)


def load_pretrained(cfg, model, ema=None, device="cuda"):
    if cfg.pretrained is None:
        return
    load_model(cfg.pretrained, model, ema=ema, device=device,
               label="Pretrained Model")
