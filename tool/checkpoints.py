"""
保存 pth
"""
import os
import torch


def save_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None,
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
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None,
        device="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = checkpoint.get("epoch", 0)
    step = checkpoint.get("step", 0)
    best_metrics = checkpoint.get("best_metrics", {})
    return epoch, step, best_metrics
