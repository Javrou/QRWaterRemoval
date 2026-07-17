from configs.base import BaseExperimentConfig


class PretrainConfig(BaseExperimentConfig):
    # ========= Dataset =========
    train_root = "synthetic_dataset/train"
    val_root = "synthetic_dataset/val"
    test_root = "synthetic_dataset/test"
    batch_size = 32

    # ========= Model =========
    use_ema = False
    ema_decay = 0.999

    # ========= Optimizer =========
    epochs = 50
    lr = 1e-4
    scheduler = "warmup_cosine"

    # ========= Log =========
    step_log = "logs/pretrain/train_step.csv"
    epoch_log = "logs/pretrain/epoch_metrics.csv"

    # ========= Checkpoint =========
    ckpt_dir = "checkpoints/pretrain"
    resume = False
    resume_path = "checkpoints/pretrain/latest.pth"
    best_zxing_path = "checkpoints/pretrain/best_zxing.pth"

    # ========= Early Stop =========
    patience = 7
    early_stop_zxing = 0.95
    min_delta_zxing = 2e-3

    # ========= Visual =========
    visual_dir = "visual/pretrain"
