from pathlib import Path

import cv2
import numpy as np


class Visualizer:
    def __init__(self, save_dir):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def to_uint8(img):

        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        img = np.squeeze(img)
        img = np.clip(img, 0, 1)

        return (img * 255).astype(np.uint8)

    def save_compare(
            self,
            inp,
            pred,
            target,
            name
    ):
        inp = self.to_uint8(inp)
        pred = self.to_uint8(pred)
        target = self.to_uint8(target)

        canvas = np.concatenate(
            [inp, pred, target],
            axis=1
        )

        cv2.imwrite(name, canvas)

    def save_batch(
            self,
            inputs,
            preds,
            targets,
            epoch=None,
            max_samples=5
    ):
        save_dir = self.save_dir
        if epoch is not None:
            save_dir = save_dir / f"epoch_{epoch}"
            save_dir.mkdir(parents=True, exist_ok=True)

        n = min(len(inputs), max_samples)
        for i in range(n):
            self.save_compare(
                inputs[i],
                preds[i],
                targets[i],
                str(save_dir / f"{i:02d}.png")
            )