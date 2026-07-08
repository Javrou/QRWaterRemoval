from torch.utils.data import DataLoader
from tool.pretrain_dataset import QRDataset
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

train_set = QRDataset(BASE_DIR / "synthetic/train", train=True)
val_set = QRDataset(BASE_DIR / "synthetic/val", train=False)
test_set = QRDataset(BASE_DIR / "synthetic/test", train=False)
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
    batch_size=24,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

