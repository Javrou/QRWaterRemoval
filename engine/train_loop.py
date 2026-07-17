import time
import torch

from utils.train_logger import StepLogger, EpochLogger


def run_training(
        cfg,
        trainer,
        evaluator,
        session,
        train_loader,
        val_loader,
        mode
):
    state = session.state

    for epoch in range(state.epoch, cfg.epochs):
        t_epoch = time.time()
        running_loss = 0.0

        for i, (inp, tgt) in enumerate(train_loader):

            t0 = time.time()

            pred, loss, grad_norm = trainer.train_step(inp, tgt)

            running_loss += loss
            state.global_step += 1
            # ------------------------------
            # Step Log
            # ------------------------------
            if (i + 1) % cfg.print_freq == 0:
                StepLogger.print(
                    epoch=epoch,
                    batch=i + 1,
                    total_batch=len(train_loader),
                    loss=loss,
                    lr=trainer.current_lr(),
                    grad_norm=grad_norm,
                    elapsed=time.time() - t0
                )
                session.step_logger.log(
                    epoch=epoch,
                    step=state.global_step,
                    loss=loss,
                    lr=trainer.current_lr(),
                    grad_norm=grad_norm,
                    zxing=None
                )
            # ------------------------------
            # ZXing
            # ------------------------------
            if (i + 1) % cfg.zxing_freq == 0:

                with torch.no_grad():
                    sr = evaluator.batch_zxing(pred)
                StepLogger.print_zxing(
                    batch=i + 1,
                    zxing=sr
                )
                session.step_logger.log(
                    epoch=epoch,
                    step=state.global_step,
                    loss=loss,
                    lr=trainer.current_lr(),
                    grad_norm=grad_norm,
                    zxing=sr
                )

        # ===========================================
        # Epoch Finished
        # ===========================================
        train_loss = running_loss / len(train_loader)
        val_metrics = evaluator.evaluate(
            loader=val_loader,
            mode=mode,
            save_visual=cfg.save_visual,
            epoch=epoch
        )
        trainer.step_scheduler(val_metrics["loss"])
        EpochLogger.print(
            epoch=epoch,
            epoch_time=time.time() - t_epoch,
            train_loss=train_loss,
            metrics=val_metrics
        )
        stop = session.finish_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_metrics=val_metrics
        )

        if stop:
            break

    print("=" * 40)
    print("Training Finished")
    print("=" * 40)
