from datasets.base_dataset import BaseQRDataset
from utils.degradation import degrade_qr

import random
from pathlib import Path


class MixedQRDataset(BaseQRDataset):

    def __init__(
            self,
            real_root,
            synthetic_root,
            synthetic_ratio=0.2,
            gray=True
    ):

        super().__init__(gray)
        self.synthetic_ratio = synthetic_ratio

        real_root = Path(real_root)
        self.real_input = real_root / "input"
        self.real_target = real_root / "target"
        self.real_files = sorted(self.real_target.glob("*.png"))

        synthetic_root = Path(synthetic_root)
        self.synthetic_target = synthetic_root / "train" / "target"
        self.synthetic_files = sorted(self.synthetic_target.glob("*.png"))

        self.length = len(self.real_files) + int(
            len(self.real_files) * synthetic_ratio
        )

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        if index < len(self.real_files):
            target_path = self.real_files[index]
            inp = self.read_img(self.real_input / target_path.name)
            target = self.read_img(target_path)

        else:
            target_path = random.choice(self.synthetic_files)
            target = self.read_img(target_path)
            inp = degrade_qr(target)

        return self.preprocess(inp), self.preprocess(target)
