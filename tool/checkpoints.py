import os
import torch


def save_checkpoint(
        path,
        model,
        ema=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        epoch=0,
        step=0,
        best_metrics=None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "best_metrics": best_metrics if best_metrics is not None else {}
    }

    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
        path,
        model,
        ema=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device="cpu"
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model"])

    if ema is not None:
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        else:
            ema.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    best_metrics = checkpoint.get("best_metrics", {})

    return epoch, step, best_metrics
