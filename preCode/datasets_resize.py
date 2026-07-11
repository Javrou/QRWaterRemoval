"""
图片尺寸重置（默认256）
"""
import cv2
import numpy as np
from pathlib import Path

input_dir = Path("../raw_data/qr_code_waterdrop/input")
target_dir = Path("../raw_data/7k_real_dataset/png_target")

out_input = Path("../raw_data/qr_code_waterdrop/input")
out_target = Path("../raw_data/7k_real_dataset/png_target")

out_input.mkdir(parents=True, exist_ok=True)
out_target.mkdir(parents=True, exist_ok=True)


def resize_keep_aspect(img, size=256):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - nh) // 2
    x_offset = (size - nw) // 2

    canvas[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
    return canvas


for f in target_dir.glob("*.png"):
    img_in = cv2.imread(str(f))
    img_tar = cv2.imread(str(target_dir / f.name))
    if img_in is None or img_tar is None:
        continue
    img_in = resize_keep_aspect(img_in)
    img_tar = resize_keep_aspect(img_tar)
    cv2.imwrite(str(out_input / f.name), img_in)
    cv2.imwrite(str(out_target / f.name), img_tar)

print("done")
