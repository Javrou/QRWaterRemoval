import torch
import cv2
import random

from pathlib import Path
from torch.utils.data import Dataset

from tool.degradation import degrade_qr


class MixedQRDataset(Dataset):
    def __init__(
        self,
        real_root,
        synthetic_root,
        synthetic_ratio=0.2
    ):
        self.synthetic_ratio = synthetic_ratio
        # Real
        real_root = Path(real_root)
        self.real_input = (
            real_root /
            "input"
        )
        self.real_target = (
            real_root /
            "target"
        )
        self.real_files = sorted(
            self.real_target.glob("*.png")
        )

        # Syntheti
        synthetic_root = Path(
            synthetic_root
        )
        self.synthetic_target = (
            synthetic_root /
            "target"
        )
        self.synthetic_files = sorted(
            self.synthetic_target.glob("*.png")
        )

        self.length = (
            len(self.real_files)
            +
            int(
                len(self.real_files)
                *
                synthetic_ratio
            )
        )

    def __len__(self):
        return self.length

    def read_img(self,path):
        img = cv2.imread(
            str(path)
        )
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        return img

    def preprocess(self,img):
        img = img.astype("float32") / 255.
        img = torch.from_numpy(img).permute(2,0,1)

        return img

    def __getitem__(self,index):
        if index < len(self.real_files):
            target_path = (self.real_files[index])
            name = target_path.name
            input_path = (self.real_input/name)
            input_img = self.read_img(input_path)
            target = self.read_img(target_path)
        else:
            syn_index = random.randint(
                0,
                len(self.synthetic_files)-1
            )
            target_path = (self.synthetic_files[syn_index])
            target = self.read_img(target_path)
            # 在线退化
            input_img = degrade_qr(target)

        input_img = self.preprocess(input_img)
        target = self.preprocess(target)

        return (
            input_img.float(),
            target.float()
        )