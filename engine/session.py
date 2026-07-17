from engine.checkpoint import save_checkpoint
from pathlib import Path


class TrainSession:

    def __init__(
            self,
            cfg,
            state,
            step_logger,
            epoch_logger,
            earlystop
    ):

        self.cfg = cfg
        self.state = state

        self.step_logger = step_logger
        self.epoch_logger = epoch_logger
        self.earlystop = earlystop

        self.best_metrics = {
            "loss": float("inf"),
            "zxing": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "binary_acc": 0.0
        }

    def finish_epoch(
            self,
            epoch,
            train_loss,
            val_metrics
    ):

        loss_improved = (
            self.best_metrics["loss"] - val_metrics["loss"]
        ) > self.cfg.min_delta_loss

        zxing_improved = (
            val_metrics["zxing"] - self.best_metrics["zxing"]
        ) > self.cfg.min_delta_zxing

        if loss_improved:
            self.best_metrics["loss"] = val_metrics["loss"]

        if zxing_improved:
            self.best_metrics["zxing"] = val_metrics["zxing"]
            self.best_metrics["psnr"] = val_metrics["psnr"]
            self.best_metrics["ssim"] = val_metrics["ssim"]
            self.best_metrics["binary_acc"] = val_metrics["binary_acc"]

        latest_metrics = {
            **val_metrics,
            "best_loss": self.best_metrics["loss"],
            "best_zxing": self.best_metrics["zxing"],
            "best_psnr": self.best_metrics["psnr"],
            "best_ssim": self.best_metrics["ssim"],
            "best_binary_acc": self.best_metrics["binary_acc"]
        }

        self.save_all(epoch, latest_metrics)

        self.epoch_logger.log(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_metrics["loss"],
            lr=self.state.optimizer.param_groups[0]["lr"],
            zxing=val_metrics["zxing"],
            psnr=val_metrics["psnr"],
            ssim=val_metrics["ssim"],
            binary_acc=val_metrics["binary_acc"],
            best_loss=self.best_metrics["loss"],
            best_zxing=self.best_metrics["zxing"],
            best_psnr=self.best_metrics["psnr"],
            best_ssim=self.best_metrics["ssim"]
        )

        return self.earlystop.step(
            epoch,
            loss_improved,
            zxing_improved,
            self.best_metrics["zxing"]
        )

    def save_all(
            self,
            epoch,
            metrics
    ):
        self.save("latest.pth", epoch, metrics)
        if metrics["loss"] == metrics["best_loss"]:
            self.save("best_loss.pth", epoch, metrics)
        if metrics["zxing"] == metrics["best_zxing"]:
            self.save("best_zxing.pth", epoch, metrics)

    def save(
            self,
            filename,
            epoch,
            metrics
    ):
        save_checkpoint(
            path=str(Path(self.cfg.ckpt_dir) / filename),
            model=self.state.model,
            ema=self.state.ema.ema if self.state.ema else None,
            optimizer=self.state.optimizer,
            scheduler=self.state.scheduler,
            scaler=self.state.scaler,
            epoch=epoch,
            step=self.state.global_step,
            metrics=metrics
        )
