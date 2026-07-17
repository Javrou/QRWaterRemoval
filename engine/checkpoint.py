import torch
import os


def save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch,
        step,
        metrics,
        ema=None
):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "metrics": metrics
    }

    if ema is not None:
        state["ema"] = ema.state_dict()

    torch.save(state, path)


def load_checkpoint(
        path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        ema=None,
        device="cpu"
):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model"]
    )

    if optimizer:
        optimizer.load_state_dict(
            ckpt["optimizer"]
        )

    if scheduler and ckpt["scheduler"]:
        scheduler.load_state_dict(
            ckpt["scheduler"]
        )

    if scaler and ckpt["scaler"]:
        scaler.load_state_dict(
            ckpt["scaler"]
        )

    if ema and "ema" in ckpt:
        ema.load_state_dict(
            ckpt["ema"]
        )

    return (
        ckpt["epoch"],
        ckpt["step"],
        ckpt.get("metrics", {})
    )
