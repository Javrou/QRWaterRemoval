import os
import cv2
from torch.utils.data import Dataset


class QRRestorationDataset(Dataset):
    def __init__(self, root_dir):
        self.input_dir = os.path.join(root_dir, "input")
        self.target_dir = os.path.join(root_dir, "target")
        self.files = sorted(os.listdir(self.input_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        input_path = os.path.join(self.input_dir, name)
        target_path = os.path.join(self.target_dir, name)
        inp = cv2.imread(input_path)
        tgt = cv2.imread(target_path)
        if inp is None or tgt is None:
            raise RuntimeError(f"Failed to read: {name}")

        # BGR -> RGB
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2RGB)

        # normalize
        inp = inp.astype("float32") / 255.0
        tgt = tgt.astype("float32") / 255.0

        # HWC -> CHW
        inp = inp.transpose(2, 0, 1)
        tgt = tgt.transpose(2, 0, 1)
        return inp, tgt



