from pathlib import Path

from datasets.base_dataset import BaseQRDataset
from utils.degradation import degrade_qr


class SyntheticQRDataset(BaseQRDataset):

    def __init__(self, root, gray=True):

        super().__init__(gray)

        self.root = Path(root)

        self.target_dir = self.root / "target"

        self.files = sorted(
            self.target_dir.glob("*.png")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        target_path = self.files[index]
        target = self.read_img(target_path)
        inp = degrade_qr(target)

        if self.augment:
            import random
            inp, target = self._augment_pair(inp, target, random.randint(0, 2**31 - 1))

        return self.preprocess(inp), self.preprocess(target)