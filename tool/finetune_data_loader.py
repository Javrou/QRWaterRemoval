from torch.utils.data import DataLoader
from tool.mixed_dataset import MixedQRDataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

train_set = MixedQRDataset(
    real_root=BASE_DIR / "real_dataset/train",
    synthetic_root=BASE_DIR / "synthetic_dataset/train",
    synthetic_ratio=0.15
)
val_set = MixedQRDataset(
    real_root=BASE_DIR / "real_dataset/val",
    synthetic_root=BASE_DIR / "synthetic_dataset/val",
    synthetic_ratio=0
)
test_set = MixedQRDataset(
    real_root=BASE_DIR / "real_dataset/test",
    synthetic_root=BASE_DIR / "synthetic_dataset/test",
    synthetic_ratio=0
)

train_loader = DataLoader(
    train_set,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

val_loader = DataLoader(
    val_set,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_set,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)