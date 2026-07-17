from configs.base import BaseExperimentConfig


class FinetuneConfig(BaseExperimentConfig):
    # ========= Dataset =========
    train_root = "data/real_dataset/train"
    val_root = "data/real_dataset/val"
    test_root = "data/real_dataset/test"
    synthetic_root = "data/synthetic_dataset"
    synthetic_ratio = 0.2
    batch_size = 16

    # ========= Model =========
    use_ema = True
    ema_decay = 0.9995

    # ========= Optimizer =========
    epochs = 30
    lr = 5e-5
    scheduler = "warmup_cosine"

    # ========= Log =========
    step_log = "logs/finetune/train_step.csv"
    epoch_log = "logs/finetune/epoch_metrics.csv"

    # ========= Checkpoint =========
    ckpt_dir = "checkpoints/finetune"
    resume = True
    resume_path = "checkpoints/finetune/latest.pth"
    pretrained = "checkpoints/pretrain/best_zxing.pth"
    best_zxing_path = "checkpoints/finetune/best_zxing.pth"

    # ========= Early Stop =========
    patience = 5
    early_stop_zxing = 0.90
    min_delta_zxing = 1e-3

    # ========= Visual =========
    visual_dir = "visual/finetune"
