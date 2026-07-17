import math
import torch.optim as optim
from torch.amp import GradScaler


class WarmupCosineLR(optim.lr_scheduler.LRScheduler):
    """Linear warmup + cosine annealing"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self):
        t = self.last_epoch
        if t < self.warmup_epochs:
            alpha = (t + 1) / self.warmup_epochs
            return [group["initial_lr"] * alpha for group in self.optimizer.param_groups]
        else:
            progress = (t - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return [self.eta_min + (group["initial_lr"] - self.eta_min) * 0.5 * (1 + math.cos(math.pi * progress))
                    for group in self.optimizer.param_groups]


def build_optimizer(model, cfg):
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    if cfg.scheduler == "warmup_cosine":
        scheduler = WarmupCosineLR(
            optimizer,
            warmup_epochs=cfg.warmup_epochs,
            total_epochs=cfg.epochs,
            eta_min=cfg.min_lr
        )

    elif cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.min_lr
        )

    elif cfg.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=cfg.lr_patience,
            threshold=1e-3,
            min_lr=cfg.min_lr
        )
    else:
        scheduler = None

    scaler = GradScaler("cuda")

    return optimizer, scheduler, scaler
