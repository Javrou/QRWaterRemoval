"""
为二维码添加模拟水滴效果
"""

import cv2
import numpy as np


def add_waterdrop(img):
    h, w = img.shape[:2]

    # 随机“水滴mask”
    mask = np.zeros((h, w), np.float32)

    for _ in range(np.random.randint(5, 12)):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        r = np.random.randint(15, 80)

        cv2.circle(mask, (x, y), r, 1, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), 15)

    # 模拟“折射位移场”
    dx = cv2.GaussianBlur((mask - 0.5), (0, 0), 20)
    dy = cv2.GaussianBlur((mask - 0.5), (0, 0), 20)

    dx = (dx * 10).astype(np.float32)
    dy = (dy * 10).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR
    )

    # 局部模糊 + 高光
    blur = cv2.GaussianBlur(warped, (0, 0), 2.5)

    alpha = mask[..., None]

    out = warped * (1 - alpha) + blur * alpha

    return out.astype(np.uint8)
