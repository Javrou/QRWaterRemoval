from engine.session import TrainSession
from engine.earlystop import EarlyStopping

from utils.train_logger import StepLogger
from utils.train_logger import EpochLogger


class TrainState:
    def __init__(self, model, optimizer, scheduler, scaler, ema):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.ema = ema
        self.epoch = 0
        self.global_step = 0


def build_session(cfg, model, optimizer, scheduler, scaler, ema):
    step_logger = StepLogger(cfg.step_log)
    epoch_logger = EpochLogger(cfg.epoch_log)

    earlystop = EarlyStopping(
        patience=cfg.patience,
        min_epochs=cfg.min_epochs,
        target_zxing=cfg.early_stop_zxing,
        min_delta_loss=cfg.min_delta_loss,
        min_delta_zxing=cfg.min_delta_zxing
    )

    state = TrainState(model, optimizer, scheduler, scaler, ema)

    session = TrainSession(
        cfg=cfg,
        state=state,
        step_logger=step_logger,
        epoch_logger=epoch_logger,
        earlystop=earlystop
    )

    return session
