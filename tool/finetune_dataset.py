import torch
import cv2

from torch.utils.data import Dataset
from pathlib import Path


class FineTuneQRDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.input_dir = self.root / "input"
        self.target_dir = self.root / "target"
        self.files = sorted(
            self.target_dir.glob("*.png")
        )

    def __len__(self):
        return len(self.files)

    def read_img(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        return img

    def preprocess(self, img):
        img = img.astype("float32") / 255.
        img = torch.from_numpy(img).unsqueeze(0)
        return img

    def __getitem__(self, index):
        target_path = self.files[index]
        name = target_path.name

        input_path = self.input_dir / name
        input_img = self.read_img(input_path)
        target = self.read_img(target_path)
        input_img = self.preprocess(input_img)
        target = self.preprocess(target)

        return (
            input_img.float(),
            target.float()
        )
