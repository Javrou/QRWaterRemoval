import torch
import cv2

from torch.utils.data import Dataset
from pathlib import Path
from tool.degradation import degrade_qr


class PretrainQRDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = Path(root)
        self.train = train

        if train:
            self.target_dir = self.root / "target"
            self.files = sorted(
                self.target_dir.glob("*.png")
            )
        else:
            self.input_dir = self.root / "input"
            self.target_dir = self.root / "target"
            self.files = sorted(
                self.target_dir.glob("*.png")
            )

    def __len__(self):
        return len(self.files)

    def read_img(self,path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess(self,img):
        img = img.astype("float32") / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __getitem__(self,index):
        target_path = self.files[index]
        if self.train:
            target = self.read_img(target_path)
            # 在线退化
            input_img = degrade_qr(target)
        else:
            name = target_path.name
            input_img = self.read_img(
                self.input_dir / name
            )
            target = self.read_img(
                target_path
            )

        input_img=self.preprocess(
            input_img
        )
        target=self.preprocess(
            target
        )
        return input_img.float(), target.float()


