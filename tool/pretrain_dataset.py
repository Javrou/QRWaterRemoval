import torch
import cv2

from torch.utils.data import Dataset
from pathlib import Path
from degradation import degrade_qr


class QRDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = Path(root)
        self.train = train
        if train:
            # 只有target
            self.files = sorted(
                self.root.glob("*.png")
            )

        else:
            # input target结构
            self.input_dir = self.root / "input"
            self.target_dir = self.root / "target"

            self.files = sorted(
                self.target_dir.glob("*.png")
            )

    def __len__(self):
        return len(self.files)

    def reamj_img(self,path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def preprocess(self,img):
        img = img.astype("float32") / 255.
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def __getitem__(self,index):
        if self.train:
            target = self.read_img(self.files[index])
            # 在线随机退化
            input_img = degrade_qr(target)
        else:
            target_path = self.files[index]
            name = target_path.name
            input_path = (
                self.input_dir /
                name
            )
            input_img = self.read_img(
                input_path
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


