# -*- coding: utf-8 -*-
"""二维码水滴干扰合成 — 微透镜物理模型

核心思路：真实水滴 = 微型凸透镜 + 曲面反射

物理效果（按重要性排序）：
1. 透镜放大/变形 — 水滴将下方 QR 模块放大扭曲（cv2.remap）
2. 边缘暗环（meniscus）— 水滴曲面边缘将光线偏折出相机视野
3. 镜面高光 — 水滴顶面反射光源形成亮斑
4. 整体微暗化 — 曲面散射降低水滴内部对比度

支持 RGB (H,W,3) 和灰度 (H,W) uint8 输入。
"""

import cv2
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
#  核心：单个水滴渲染（微透镜模型）
# ═══════════════════════════════════════════════════════════════════════════

def _single_droplet(img, cx, cy, rx, ry, angle,
                    edge_dark=0.55, inner_dark=0.10,
                    lens_strength=0.08, highlight=True,
                    seed=0):
    """在图像上渲染一个水滴（微透镜模型）

    Parameters
    ----------
    rx, ry : float — 椭圆长/短半轴
    angle : float — 椭圆旋转角（弧度）
    edge_dark : float (0-1) — meniscus 边缘暗化强度
    inner_dark : float (0-1) — 水滴内部整体暗化
    lens_strength : float — 透镜放大位移强度
    highlight : bool — 是否添加镜面高光
    """
    h, w = img.shape[:2]
    is_gray = (len(img.shape) == 2)
    img_f = img.astype(np.float32)

    # ---- 坐标网格（相对于水滴中心）----
    y, x = np.meshgrid(np.arange(w), np.arange(h))
    dx = x - cx
    dy = y - cy

    # 旋转到椭圆坐标系
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    # 归一化距离 (椭圆内 < 1)
    d = np.sqrt((dx_rot / rx)**2 + (dy_rot / ry)**2)

    # ---- 水滴软边缘 mask（smoothstep）----
    edge_width = 0.10
    t = np.clip((1.0 - d) / edge_width, 0, 1)
    mask = (t * t * (3.0 - 2.0 * t)).astype(np.float32)

    # ---- 1) 透镜放大/变形（核心物理效果）----
    # 凸透镜：水滴中心像素保持原位，中间区域向外放大
    # 位移 = strength * d * (1-d) * rx（抛物线形状，在 d=0.5 处最大）
    displacement = lens_strength * d * np.clip(1.0 - d, 0, 1) * rx
    displacement *= mask

    disp_norm = d + 1e-8
    disp_x = displacement * dx_rot / disp_norm
    disp_y = displacement * dy_rot / disp_norm

    # 回到原始坐标系
    disp_x_final = disp_x * cos_a - disp_y * sin_a
    disp_y_final = disp_x * sin_a + disp_y * cos_a

    map_x = (x + disp_x_final).astype(np.float32)
    map_y = (y + disp_y_final).astype(np.float32)

    lensed = cv2.remap(img_f, map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ---- 2) 边缘暗环（meniscus）----
    # 水滴曲面边缘（d ≈ 0.82-0.96）光线偏折出相机，产生暗环
    meniscus = np.exp(-0.5 * ((d - 0.88) / 0.07)**2) * mask
    meniscus = np.clip(meniscus, 0, 1)

    # 内部微弱暗化
    inner = np.clip(mask - meniscus * 0.5, 0, 1)

    total_dark = 1.0 - (meniscus * edge_dark + inner * inner_dark)
    total_dark = np.clip(total_dark, 0, 1)

    if is_gray:
        darkened = lensed * total_dark
    else:
        darkened = lensed * np.dstack([total_dark] * 3)

    # ---- 3) 镜面高光 ----
    if highlight:
        rng = np.random.RandomState(seed)
        # 光源位置：水滴左上象限
        light_dx = dx_rot + rx * rng.uniform(0.1, 0.35)
        light_dy = dy_rot - ry * rng.uniform(0.05, 0.25)
        hl_d = np.sqrt((light_dx / (rx * 0.22))**2 +
                        (light_dy / (ry * 0.10))**2)
        hl = np.exp(-hl_d**2) * mask
        hl = np.clip(hl, 0, 1)
        hl_intensity = rng.uniform(0.10, 0.28) * 255

        if is_gray:
            darkened += hl * hl_intensity
        else:
            for c in range(3):
                darkened[:, :, c] += hl * hl_intensity

    # ---- 4) 合成(mask需与通道匹配) ----
    if is_gray:
        result = img_f * (1.0 - mask) + darkened * mask
    else:
        mask_3 = np.dstack([mask] * 3)
        result = img_f * (1.0 - mask_3) + darkened * mask_3

    # 水滴内部轻微散射模糊
    blur_k = max(3, int(min(rx, ry) * 0.06))
    if blur_k % 2 == 0:
        blur_k += 1
    if blur_k >= 3:
        blurred = cv2.GaussianBlur(result, (blur_k, blur_k),
                                   sigmaX=blur_k * 0.28)
        if is_gray:
            result = result * (1.0 - mask * 0.30) + blurred * mask * 0.30
        else:
            result = result * (1.0 - mask_3 * 0.30) + blurred * mask_3 * 0.30

    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
#  批量水滴生成
# ═══════════════════════════════════════════════════════════════════════════

def add_waterdrop(img, num_drops=None, radius_range=(6, 55), seed=None):
    """在图像上随机散布水滴

    每个水滴：近圆形椭圆（aspect 0.7-1.0）+ 微透镜效果
    """
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    result = img.copy()

    if num_drops is None:
        num_drops = rng.randint(4, 10)

    for i in range(num_drops):
        r = rng.randint(*radius_range)
        aspect = rng.uniform(0.7, 1.0)
        rx = r
        ry = r * aspect
        angle = rng.uniform(0, 2 * np.pi)

        # 允许水滴中心出现在图像85%区域（含边缘）
        margin = max(0, int(min(rx, ry) * 0.15))
        if margin * 2 >= min(h, w):
            continue

        cx = rng.randint(margin, max(margin + 1, w - margin))
        cy = rng.randint(margin, max(margin + 1, h - margin))

        edge_dk = rng.uniform(0.40, 0.65)
        inner_dk = rng.uniform(0.05, 0.18)
        lens = rng.uniform(0.04, 0.12)
        do_hl = rng.random() < 0.55

        result = _single_droplet(
            result, cx, cy, rx, ry, angle,
            edge_dark=edge_dk, inner_dark=inner_dk,
            lens_strength=lens, highlight=do_hl,
            seed=rng.randint(0, 2**31))

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  辅助后处理
# ═══════════════════════════════════════════════════════════════════════════

def random_blur(img, prob=0.3, seed=None):
    if np.random.RandomState(seed).random() > prob:
        return img
    rng = np.random.RandomState(seed)
    k = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), sigmaX=rng.uniform(0.4, 1.2))


def random_noise(img, prob=0.4, intensity_range=(2, 10), seed=None):
    if np.random.RandomState(seed).random() > prob:
        return img
    rng = np.random.RandomState(seed)
    noise = rng.randn(*img.shape).astype(np.float32)
    intensity = rng.uniform(*intensity_range)
    result = img.astype(np.float32) + noise * intensity
    return np.clip(result, 0, 255).astype(np.uint8)


def random_brightness(img, prob=0.4, range_=(-12, 20), seed=None):
    if np.random.RandomState(seed).random() > prob:
        return img
    rng = np.random.RandomState(seed)
    delta = rng.uniform(*range_)
    result = img.astype(np.float32) + delta
    return np.clip(result, 0, 255).astype(np.uint8)


def random_contrast(img, prob=0.35, factor_range=(0.88, 1.15), seed=None):
    if np.random.RandomState(seed).random() > prob:
        return img
    rng = np.random.RandomState(seed)
    factor = rng.uniform(*factor_range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    result = (img.astype(np.float32) - mean) * factor + mean
    return np.clip(result, 0, 255).astype(np.uint8)


def random_vignette(img, prob=0.2, seed=None):
    if np.random.RandomState(seed).random() > prob:
        return img
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    is_gray = (len(img.shape) == 2)
    yc, xc = np.meshgrid(np.arange(w), np.arange(h))
    r = np.sqrt((xc - w/2)**2 + (yc - h/2)**2) / (np.sqrt((w/2)**2 + (h/2)**2))
    vignette = 1 - rng.uniform(0.1, 0.35) * r
    vignette = np.clip(vignette, 0, 1)
    if is_gray:
        result = img.astype(np.float32) * vignette
    else:
        result = img.astype(np.float32) * np.dstack([vignette] * 3)
    return np.clip(result, 0, 255).astype(np.uint8)


def random_perspective(img, prob=0.15, max_shift=0.03, seed=None):
    if np.random.RandomState(seed).random() > prob:
        return img
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    shift = max_shift * min(h, w)
    dst_pts = src_pts + rng.uniform(-shift, shift, (4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════════

def degrade_qr(img, seed=None):
    """主退化函数 — 微透镜物理模型模拟真实水滴干扰

    物理流水线：
    1. 每个水滴 = 凸透镜 → cv2.remap 放大/扭曲 QR 模块
    2. meniscus 边缘暗环 → 曲面边缘光线偏折出相机
    3. 镜面高光 → 水滴顶面的亮反射
    4. 轻微后处理 → 模糊/噪声/亮度/对比度/暗角

    支持 RGB (H,W,3) 和灰度 (H,W) uint8 输入。
    """
    is_gray = (len(img.shape) == 2)
    if img.dtype != np.uint8:
        raise ValueError("img must be uint8")
    if not is_gray and (len(img.shape) != 3 or img.shape[2] != 3):
        raise ValueError("img must be uint8 RGB (H,W,3) or grayscale (H,W)")

    rng = np.random.RandomState(seed)

    result = add_waterdrop(img, seed=rng.randint(0, 2**31))
    result = random_blur(result, seed=rng.randint(0, 2**31))
    result = random_noise(result, seed=rng.randint(0, 2**31))
    result = random_brightness(result, seed=rng.randint(0, 2**31))
    result = random_contrast(result, seed=rng.randint(0, 2**31))
    result = random_vignette(result, seed=rng.randint(0, 2**31))
    result = random_perspective(result, seed=rng.randint(0, 2**31))
    return result
