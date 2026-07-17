import torch
from torch.amp import autocast


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        loss_fn,
        device,
        ema=None,
        grad_clip=4
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.loss_fn = loss_fn
        self.device = device
        self.ema = ema
        self.grad_clip = grad_clip

    def train_step(self, inp, tgt):

        self.model.train()

        inp = inp.to(self.device, non_blocking=True)
        tgt = tgt.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):

            pred = self.model(inp)

            pred = pred.clamp(0, 1)

            loss = self.loss_fn(pred, tgt)

        self.scaler.scale(loss).backward()

        self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip
        ).item()

        self.scaler.step(self.optimizer)

        self.scaler.update()

        if self.ema is not None:
            self.ema.update(self.model)

        return (
            pred.detach(),
            loss.item(),
            grad_norm
        )

    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def step_scheduler(self, metric=None):

        if self.scheduler is None:
            return

        if isinstance(
            self.scheduler,
            torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            self.scheduler.step(metric)
        else:
            self.scheduler.step()