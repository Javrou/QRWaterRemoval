import cv2
import numpy as np
import torch

from pathlib import Path
from torch.utils.data import Dataset


class BaseQRDataset(Dataset):

    def __init__(self, gray=True):
        self.gray = gray
        self.augment = True

    def _augment_pair(self, inp, target, seed):
        rng = np.random.RandomState(seed)
        h, w = inp.shape[:2]
        angle = rng.uniform(-12, 12)
        scale = rng.uniform(0.85, 1.01)
        flip_h = rng.random() < 0.5
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        def _apply(img):
            if img is None:
                return None
            img = cv2.warpAffine(img, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            if flip_h:
                img = cv2.flip(img, 1)
            return img

        return _apply(inp), _apply(target)

    def read_img(self, path):
        if self.gray:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def preprocess(self, img):

        img = img.astype("float32") / 255.
        if self.gray:
            img = torch.from_numpy(img).unsqueeze(0)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)

        return img.float()