import random
import cv2
import numpy as np


def add_waterdrop(img, severity="medium"):
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    if severity == "light":
        drop_num = random.randint(4, 7)
        radius_range = (18, 45)
        refraction = random.uniform(9, 14)
        blur_sigma = random.uniform(1.3, 2.5)
        drop_strength = random.uniform(0.4, 0.7)
        blur_strength = random.uniform(0.45, 0.8)
        specular_strength = random.uniform(0.1, 0.22)

    elif severity == "heavy":
        drop_num = random.randint(10, 15)
        radius_range = (35, 80)
        refraction = random.uniform(17, 25)
        blur_sigma = random.uniform(2.5, 4.0)
        drop_strength = random.uniform(0.6, 0.85)
        blur_strength = random.uniform(0.7, 1.0)
        specular_strength = random.uniform(0.18, 0.35)

    else:
        drop_num = random.randint(7, 12)
        radius_range = (30, 70)
        refraction = random.uniform(13, 20)
        blur_sigma = random.uniform(1.8, 3.0)
        drop_strength = random.uniform(0.55, 0.85)
        blur_strength = random.uniform(0.5, 0.92)
        specular_strength = random.uniform(0.12, 0.28)

    # water mask
    mask = np.zeros((h, w), dtype=np.float32)

    for _ in range(drop_num):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r = random.randint(radius_range[0], radius_range[1])
        cv2.circle(mask, (x, y), r, 1, -1)

    mask = cv2.GaussianBlur(mask, (0, 0), 5)

    # refraction
    distortion = cv2.GaussianBlur(mask - 0.5, (0, 0), 15)

    dx = distortion * refraction * random.uniform(-1, 1)
    dy = distortion * refraction * random.uniform(-1, 1)

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))

    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    warped = cv2.remap(
        img_f,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    # local blur

    blur = cv2.GaussianBlur(
        warped,
        (0, 0),
        blur_sigma
    )

    water_alpha = mask[..., None] * drop_strength
    blur_alpha = water_alpha * blur_strength

    out = warped * (1 - blur_alpha) + blur * blur_alpha

    # specular highlight

    edge = cv2.Laplacian(
        mask,
        cv2.CV_32F
    )

    edge = np.abs(edge)

    edge = cv2.GaussianBlur(
        edge,
        (0, 0),
        3
    )

    if edge.max() > 0:
        edge = edge / edge.max()

    edge = edge[..., None]

    specular_alpha = edge * specular_strength

    out = out * (1 - specular_alpha) + 255 * specular_alpha

    # brightness fluctuation

    brightness = random.uniform(0.85, 1.15)

    out = (
            out * (1 - water_alpha * 0.2)
            +
            out * brightness * water_alpha * 0.2
    )

    return np.clip(out, 0, 255).astype(np.uint8)


def random_blur(img):
    if random.random() < 0.4:
        k = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)
        img = cv2.GaussianBlur(
            img,
            (k, k),
            sigma
        )

    return img


def random_noise(img):
    if random.random() < 0.3:
        noise = np.random.normal(
            0,
            random.uniform(3, 15),
            img.shape
        )

        img = np.clip(
            img + noise,
            0,
            255
        )

    return img


def random_brightness(img):
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)

        img = np.clip(
            img * factor,
            0,
            255
        )

    return img


def random_contrast(img):
    if random.random() < 0.5:
        factor = random.uniform(0.6, 1.5)
        img = img.astype(np.float32)

        mean = np.float32(np.mean(img))

        img = np.clip(
            (img - mean) * factor + mean,
            0,
            255
        )

    return img.astype(np.float32)


def add_highlight(img):
    if random.random() < 0.35:
        h, w, _ = img.shape
        img = img.astype(np.float32)

        overlay = np.zeros_like(
            img,
            dtype=np.float32
        )

        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        radius = random.randint(10, 50)

        cv2.circle(
            overlay,
            (x, y),
            radius,
            (255, 255, 255),
            -1
        )

        alpha = random.uniform(
            0.08,
            0.35
        )

        img = cv2.addWeighted(
            img,
            1 - alpha,
            overlay,
            alpha,
            0,
            dtype=cv2.CV_32F
        )

    return img.astype(np.float32)


def jpeg_compress(img):
    if random.random() < 0.25:
        quality = random.randint(
            35,
            90
        )

        img_uint8 = np.clip(
            img,
            0,
            255
        ).astype(np.uint8)

        _, enc = cv2.imencode(
            ".jpg",
            img_uint8,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )

        img = cv2.imdecode(
            enc,
            cv2.IMREAD_COLOR
        )

    return img


def degrade_qr(img):
    img = img.copy().astype(np.float32)

    # severity sampling

    p = random.random()

    if p < 0.2:
        severity = "light"

    elif p < 0.75:
        severity = "medium"

    else:
        severity = "heavy"

    img = add_waterdrop(
        img.astype(np.uint8),
        severity
    )

    img = img.astype(np.float32)

    img = random_blur(img)
    img = random_noise(img)
    img = random_brightness(img)
    img = random_contrast(img)
    img = add_highlight(img)
    img = jpeg_compress(img)

    return np.clip(
        img,
        0,
        255
    ).astype(np.uint8)
