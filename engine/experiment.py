import random
import numpy as np
import torch

from builders.model import build_model
from builders.optimizer import build_optimizer
from builders.session import build_session
from builders.resume import resume_training, load_model, load_pretrained

from engine.trainer import Trainer
from engine.evaluator import Evaluator
from engine.train_loop import run_training

from utils.ema import ModelEMA

from loss import compute_loss


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_experiment(cfg_cls, build_loader_fn, mode, use_ema=False):
    # ==========================================
    # Config & Seed
    # ==========================================
    cfg = cfg_cls()
    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Torch  : {torch.__version__}")

    # ==========================================
    # Data
    # ==========================================
    train_loader, val_loader, test_loader = build_loader_fn(cfg)

    # ==========================================
    # Model
    # ==========================================
    model = build_model(cfg, device)
    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )

    # ==========================================
    # EMA & Pretrained (finetune only)
    # ==========================================
    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=cfg.ema_decay)
        load_pretrained(cfg, model, ema=ema, device=device)

    # ==========================================
    # Optimizer
    # ==========================================
    optimizer, scheduler, scaler = build_optimizer(model, cfg)

    # =====================================
    # Loss
    # =====================================
    loss_fn = lambda pred, tgt: compute_loss(pred, tgt, mode=mode)

    # ==========================================
    # Trainer
    # ==========================================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loss_fn=loss_fn,
        device=device,
        ema=ema
    )

    # ==========================================
    # Evaluator
    # ==========================================
    evaluator = Evaluator(
        model=model,
        device=device,
        visual_dir=cfg.visual_dir
    )

    # ==========================================
    # Session
    # ==========================================
    session = build_session(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema
    )

    # ==========================================
    # Resume
    # ==========================================
    start_epoch, global_step, best_metrics = resume_training(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        device=device
    )

    session.state.epoch = start_epoch
    session.state.global_step = global_step
    session.best_metrics = best_metrics

    # ==========================================
    # Training
    # ==========================================
    run_training(
        cfg=cfg,
        trainer=trainer,
        evaluator=evaluator,
        session=session,
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode
    )

    # ==========================================
    # Test
    # ==========================================
    print("\nTesting best model...")
    load_model(
        cfg.best_zxing_path,
        model,
        ema=ema,
        device=device
    )
    metrics = evaluator.evaluate(
        loader=test_loader,
        mode=mode,
        save_visual=True,
        epoch="test"
    )

    print("=" * 40)
    print("Test Result")
    print(f"Loss       : {metrics['loss']:.6f}")
    print(f"ZXing      : {metrics['zxing']:.4f}")
    print(f"PSNR       : {metrics['psnr']:.4f}")
    print(f"SSIM       : {metrics['ssim']:.4f}")
    print(f"Binary Acc : {metrics['binary_acc']:.4f}")
    print("=" * 40)
