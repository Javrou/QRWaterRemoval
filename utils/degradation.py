# -*- coding: utf-8 -*-
"""二维码水滴干扰合成模块

基于 real_dataset 的真实水滴特征进行模拟：
- 不规则有机形状（非完美圆形，支持椭圆变形）
- 软羽化边缘 + 半透明质感
- 镜面高光（亮反射点）
- 暗化为主效果（强于折射，匹配真实水滴）
- 水滴覆盖 31-57% 图像，暗化均值 ~50/255
- 水流拖尾痕迹（水滴流淌后留下的细长暗痕）
- 水膜（大面积不均匀薄水层暗化）
- 干涸残留环（蒸发后的环形痕迹）

支持 RGB (H,W,3) 和灰度 (H,W) 两种输入格式，
兼容 pretrain_dataset.py（RGB）和 mixed_dataset.py（灰度）。
"""

import cv2
import numpy as np


def _perlin_noise(shape, scales=(4, 8, 16, 32), seed=None):
    """多层 Perlin-like 噪声，用于生成有机纹理"""
    if seed is not None:
        np.random.seed(seed)
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    total_amp = 0.0
    for s in scales:
        nh = max(2, (h + s - 1) // s)
        nw = max(2, (w + s - 1) // s)
        octave = np.random.randn(nh, nw).astype(np.float32)
        octave = cv2.resize(octave, (w, h), interpolation=cv2.INTER_LINEAR)
        noise += amp * octave
        total_amp += amp
        amp *= 0.5
    noise /= total_amp
    return noise


def _irregular_drop_mask(shape, center, radius, aspect_ratio=None, angle=None,
                         noise_strength=0.3, num_verts_range=(10, 20)):
    """生成不规则水滴 mask

    通过随机多边形 + Perlin 噪声边缘变形来模拟有机水滴形状。
    支持椭圆变形（aspect_ratio + angle），避免所有水滴都是正圆。
    """
    h, w = shape
    rng = np.random
    n_verts = rng.randint(*num_verts_range)
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    angles += rng.uniform(-0.2, 0.2, size=n_verts)

    # 半径扰动 — 产生有机的不规则边缘
    radii = radius * (1.0 + noise_strength * rng.uniform(-1, 1, size=n_verts))
    radii = np.clip(radii, radius * 0.5, radius * 1.6)

    # 椭圆变形参数
    if aspect_ratio is None:
        aspect_ratio = rng.uniform(0.55, 1.0)
    if angle is None:
        angle = rng.uniform(0, 2 * np.pi)

    rx = radii * np.cos(angles) * aspect_ratio
    ry = radii * np.sin(angles)
    rot_x = rx * np.cos(angle) - ry * np.sin(angle)
    rot_y = rx * np.sin(angle) + ry * np.cos(angle)

    pts = np.column_stack([
        center[0] + rot_x,
        center[1] + rot_y
    ]).astype(np.float32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32).reshape((-1, 1, 2))], 255)

    # Perlin 噪声边缘变形 — 让水滴形状更自然
    noise = _perlin_noise((h, w),
        scales=(max(4, radius // 8), max(8, radius // 4), max(12, radius // 2)),
        seed=rng.randint(0, 2**31))
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=radius * 0.2)

    mask_float = mask.astype(np.float32) / 255.0
    mask_float += noise * 0.2 * mask_float
    mask_float = np.clip(mask_float, 0, 1)

    # 羽化边缘（匹配真实水滴的软边缘，梯度 ~32-36）
    blur_sigma = max(2.0, radius * 0.07)
    mask_float = cv2.GaussianBlur(mask_float, (0, 0), sigmaX=blur_sigma)

    return mask_float


def _water_streak_mask(shape, start, end, width):
    """水流拖尾痕迹 — 水滴向下流淌留下的细长暗痕"""
    h, w = shape
    y_coords, x_coords = np.meshgrid(np.arange(w), np.arange(h))

    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = np.sqrt(dx**2 + dy**2) + 1e-6
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux  # 法线方向

    # 沿流向的距离
    along = (x_coords - sx) * ux + (y_coords - sy) * uy
    # 到流向线的垂直距离
    across = np.abs((x_coords - sx) * nx + (y_coords - sy) * ny)

    streak = np.exp(-0.5 * (across / (width * 0.5))**2)
    streak *= np.clip(1.0 - along / length, 0, 1)   # 尾部渐隐
    streak *= np.clip(along / (width * 0.3), 0, 1)   # 起始渐入
    streak = np.clip(streak, 0, 1)

    return streak.astype(np.float32)


def _water_film_mask(shape, seed=None):
    """大面积不均匀水膜 — 模拟表面薄水层"""
    h, w = shape
    noise = _perlin_noise((h, w), scales=(32, 64, 128), seed=seed)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=20)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    return np.clip(noise * 0.4, 0, 1).astype(np.float32)


def _residue_ring(shape, center, radius):
    """干涸残留环 — 水滴蒸发后留下的环形暗痕"""
    h, w = shape
    y_coords, x_coords = np.meshgrid(np.arange(w), np.arange(h))
    dist = np.sqrt((x_coords - center[0])**2 + (y_coords - center[1])**2)

    ring_width = radius * 0.15
    ring_center = radius * 0.8
    ring = np.exp(-0.5 * ((dist - ring_center) / ring_width)**2)
    return np.clip(ring * np.random.uniform(0.12, 0.30), 0, 1).astype(np.float32)


def _apply_darkening(img, mask, opacity, darken_factor=0.55):
    """对水滴区域施加暗化效果（兼容灰度和 RGB）"""
    is_gray = (len(img.shape) == 2)
    result = img.astype(np.float32)
    mask_dark = mask * opacity * darken_factor
    if is_gray:
        result = result * (1.0 - mask_dark)
    else:
        result = result * (1.0 - np.dstack([mask_dark] * 3))
    return np.clip(result, 0, 255).astype(np.uint8)


def _add_specular_highlights(img, mask, center, radius):
    """添加镜面高光 — 水滴表面亮反射（兼容灰度和 RGB）"""
    h, w = img.shape[:2]
    is_gray = (len(img.shape) == 2)
    y_coords, x_coords = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = center

    rx = radius * np.random.uniform(0.2, 0.4)
    ry = radius * np.random.uniform(0.1, 0.25)
    angle = np.random.uniform(0, 2 * np.pi)

    dx = x_coords - cx
    dy = y_coords - cy
    rot_x = dx * np.cos(-angle) - dy * np.sin(-angle)
    rot_y = dx * np.sin(-angle) + dy * np.cos(-angle)

    highlight = np.exp(-((rot_x / rx)**2 + (rot_y / ry)**2))
    highlight *= mask
    highlight = np.clip(highlight, 0, 1)

    intensity = np.random.uniform(0.08, 0.25) * 255
    result = img.astype(np.float32)
    if is_gray:
        result += highlight * intensity
    else:
        for c in range(3):
            result[:, :, c] += highlight * intensity
    return np.clip(result, 0, 255).astype(np.uint8)


def add_waterdrop(img, num_drops=None, radius_range=(8, 55),
                  opacity_range=(0.35, 0.75)):
    """主水滴添加函数

    模拟真实水滴的四重效果：
    1. 不规则有机形状水滴（暗化为主）
    2. 镜面高光（亮反射斑）
    3. 水流拖尾痕迹（概率 25%）
    4. 干涸残留环（概率 15%）

    兼容灰度和 RGB 输入。
    """
    h, w = img.shape[:2]
    is_gray = (len(img.shape) == 2)
    result = img.astype(np.float32)

    if num_drops is None:
        num_drops = np.random.randint(5, 20)

    for _ in range(num_drops):
        radius = np.random.randint(*radius_range)
        margin = radius + 10
        if margin * 2 >= min(h, w):
            continue

        cx = np.random.randint(margin, w - margin)
        cy = np.random.randint(margin, h - margin)

        # 1) 生成不规则水滴 mask
        aspect_ratio = np.random.uniform(0.55, 1.0)
        angle = np.random.uniform(0, 2 * np.pi)
        mask = _irregular_drop_mask((h, w), (cx, cy), radius,
                                    aspect_ratio, angle)

        # 2) 暗化（核心效果）
        opacity = np.random.uniform(*opacity_range)
        darken = np.random.uniform(0.45, 0.70)
        result = _apply_darkening(result, mask, opacity, darken).astype(np.float32)

        # 3) 边缘暗化（meniscus / 表面张力边缘）
        edge_k = max(2, int(radius * 0.10))
        edge = cv2.Canny((mask * 255).astype(np.uint8), 15, 60)
        edge = cv2.dilate(edge, np.ones((edge_k, edge_k), np.uint8))
        edge_f = edge.astype(np.float32) / 255.0
        edge_dark = 1.0 - edge_f * np.random.uniform(0.30, 0.60)
        if is_gray:
            result *= edge_dark
        else:
            result *= np.dstack([edge_dark] * 3)
        result = np.clip(result, 0, 255)

        # 4) 镜面高光
        if np.random.random() < 0.6:
            result = _add_specular_highlights(
                np.clip(result, 0, 255).astype(np.uint8),
                mask, (cx, cy), radius
            ).astype(np.float32)

        # 5) 水流拖尾（概率 25%）
        if np.random.random() < 0.25:
            trail_len = np.random.uniform(radius * 0.5, radius * 2.5)
            trail_angle = np.random.uniform(np.pi * 0.3, np.pi * 0.7)
            ex = np.clip(cx + trail_len * np.cos(trail_angle), 0, w - 1)
            ey = np.clip(cy + trail_len * np.sin(trail_angle), 0, h - 1)
            streak = _water_streak_mask((h, w), (cx, cy), (ex, ey),
                                        radius * 0.22)
            result = _apply_darkening(
                np.clip(result, 0, 255).astype(np.uint8),
                streak, np.random.uniform(0.15, 0.35), 0.5
            ).astype(np.float32)

        # 6) 干涸残留环（概率 15%）
        if np.random.random() < 0.15:
            ring_r = radius * np.random.uniform(0.55, 0.85)
            ring = _residue_ring((h, w), (cx, cy), ring_r)
            if is_gray:
                result = result * (1.0 - ring)
            else:
                result = result * (1.0 - np.dstack([ring] * 3))
            result = np.clip(result, 0, 255)

    return np.clip(result, 0, 255).astype(np.uint8)


def _add_background_dirt(img, num_spots=8):
    """添加背景污渍 — 小暗斑（兼容灰度和 RGB）"""
    h, w = img.shape[:2]
    is_gray = (len(img.shape) == 2)
    result = img.astype(np.float32)

    for _ in range(num_spots):
        radius = np.random.randint(2, 10)
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        y_coords, x_coords = np.meshgrid(np.arange(w), np.arange(h))
        dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        spot = np.exp(-dist**2 / (2 * (radius * 0.7)**2))
        dirt_val = np.random.uniform(0.75, 0.95)
        if is_gray:
            result = result * (1 - spot * 0.3) + result * spot * 0.3 * dirt_val
        else:
            spot_3 = np.dstack([spot] * 3)
            result = result * (1 - spot_3 * 0.3) + result * spot_3 * 0.3 * dirt_val

    return np.clip(result, 0, 255).astype(np.uint8)


def random_blur(img, prob=0.4):
    if np.random.random() > prob:
        return img
    k = np.random.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), sigmaX=np.random.uniform(0.5, 1.5))


def random_noise(img, prob=0.5, intensity_range=(2, 12)):
    if np.random.random() > prob:
        return img
    noise = np.random.randn(*img.shape).astype(np.float32)
    intensity = np.random.uniform(*intensity_range)
    result = img.astype(np.float32) + noise * intensity
    return np.clip(result, 0, 255).astype(np.uint8)


def random_brightness(img, prob=0.5, range_=(-15, 25)):
    if np.random.random() > prob:
        return img
    delta = np.random.uniform(*range_)
    result = img.astype(np.float32) + delta
    return np.clip(result, 0, 255).astype(np.uint8)


def random_contrast(img, prob=0.4, factor_range=(0.85, 1.2)):
    if np.random.random() > prob:
        return img
    factor = np.random.uniform(*factor_range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    result = (img.astype(np.float32) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def random_vignette(img, prob=0.25):
    if np.random.random() > prob:
        return img
    h, w = img.shape[:2]
    is_gray = (len(img.shape) == 2)
    y_coords, x_coords = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = w / 2, h / 2
    r = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2) / (np.sqrt(cx**2 + cy**2))
    vignette = 1 - np.random.uniform(0.15, 0.4) * r
    vignette = np.clip(vignette, 0, 1)
    if is_gray:
        result = img.astype(np.float32) * vignette
    else:
        result = img.astype(np.float32) * np.dstack([vignette] * 3)
    return np.clip(result, 0, 255).astype(np.uint8)


def random_perspective(img, prob=0.2, max_shift=0.03):
    if np.random.random() > prob:
        return img
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    shift = max_shift * min(h, w)
    dst_pts = src_pts + np.random.uniform(-shift, shift, (4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return result


def degrade_qr(img):
    """主退化函数 — 模拟真实水滴干扰

    支持 RGB (H,W,3) 和灰度 (H,W) 输入。

    模拟流水线（匹配 real_dataset 的水滴特征）：
    1. 水膜 — 大面积薄水层造成的轻微不均匀暗化
    2. 不规则水滴 — 多重效果：暗化 + 高光 + 拖尾 + 残留环
    3. 背景污渍 — 小暗斑
    4. 轻度后处理：模糊 / 噪声 / 亮度 / 对比度 / 暗角 / 透视
    """
    is_gray = (len(img.shape) == 2)
    if img.dtype != np.uint8:
        raise ValueError("img must be uint8 image")
    if not is_gray and (len(img.shape) != 3 or img.shape[2] != 3):
        raise ValueError("img must be uint8 RGB (H,W,3) or grayscale (H,W)")

    # 1. 水膜 — 大面积薄水层造成轻微不均匀暗化
    result = img.astype(np.float32)
    film = _water_film_mask(img.shape[:2])
    result = _apply_darkening(result, film,
                              np.random.uniform(0.10, 0.35), 0.5).astype(np.float32)

    # 2. 水滴主效果 — 暗化 + 高光 + 拖尾 + 残留环
    result = add_waterdrop(
        np.clip(result, 0, 255).astype(np.uint8)
    ).astype(np.float32)

    # 3. 背景污渍
    if np.random.random() < 0.3:
        result = _add_background_dirt(
            np.clip(result, 0, 255).astype(np.uint8)
        ).astype(np.float32)

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 4. 轻度后处理
    result = random_blur(result, prob=0.35)
    result = random_noise(result, prob=0.4)
    result = random_brightness(result, prob=0.4)
    result = random_contrast(result, prob=0.3)
    result = random_vignette(result, prob=0.25)
    result = random_perspective(result, prob=0.15)

    return result
