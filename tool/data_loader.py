from torch.utils.data import DataLoader
from dataset import QRRestorationDataset

train_set = QRRestorationDataset("../real_datasets/train")
val_set = QRRestorationDataset("../real_datasets/val")
test_set = QRRestorationDataset("../real_datasets/test")


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
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

