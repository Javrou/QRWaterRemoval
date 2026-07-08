"""
保存pth
"""
import os
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best_zxing):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "best_zxing": best_zxing
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])

    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt["epoch"], ckpt["step"], ckpt.get("best_loss", 1e9)
