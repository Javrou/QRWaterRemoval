from torch.utils.data import DataLoader

from datasets.synthetic_dataset import SyntheticQRDataset
from datasets.paired_dataset import PairedQRDataset


def _loader(dataset, cfg, shuffle):

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        drop_last=shuffle
    )


def build_pretrain_loader(cfg):

    train = SyntheticQRDataset(
        cfg.train_root,
        cfg.gray
    )

    val = PairedQRDataset(
        cfg.val_root,
        cfg.gray
    )

    test = PairedQRDataset(
        cfg.test_root,
        cfg.gray
    )

    return (
        _loader(train, cfg, True),
        _loader(val, cfg, False),
        _loader(test, cfg, False)
    )


def build_mixed_loader(cfg):
    from datasets.mixed_dataset import MixedQRDataset

    train = MixedQRDataset(
        cfg.train_root,
        cfg.synthetic_root,
        cfg.synthetic_ratio,
        cfg.gray
    )

    val = PairedQRDataset(
        cfg.val_root,
        cfg.gray
    )

    test = PairedQRDataset(
        cfg.test_root,
        cfg.gray
    )

    return (
        _loader(train, cfg, True),
        _loader(val, cfg, False),
        _loader(test, cfg, False)
    )
