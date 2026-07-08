"""
生成qr code
"""
import os
import random
import string
import csv
from PIL import Image
import qrcode

from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H
)

SAVE_DIR = "../synthetic_dataset/train/target"

CSV_PATH = "../synthetic_dataset/train.csv"

NUM_IMAGES = 15000
SIZE = 256
SEED = 202607081
random.seed(SEED)
os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

ECC_MAP = {
    ERROR_CORRECT_L: "L",
    ERROR_CORRECT_M: "M",
    ERROR_CORRECT_Q: "Q",
    ERROR_CORRECT_H: "H"
}


def random_ecc():
    r = random.random()
    if r < 0.1:
        return ERROR_CORRECT_L
    elif r < 0.4:
        return ERROR_CORRECT_M
    elif r < 0.8:
        return ERROR_CORRECT_Q
    else:
        return ERROR_CORRECT_H


def random_number():
    length = random.randint(8, 20)
    return ''.join(
        random.choice(
            string.digits
        )
        for _ in range(length)
    )


def random_text():
    chars = (
            string.ascii_uppercase
            +
            string.digits
    )

    length = random.randint(10, 30)

    return ''.join(
        random.choice(chars)
        for _ in range(length)
    )


def random_url():
    return (
            "https://qr.test/"
            +
            str(
                random.randint(
                    100000,
                    99999999
                )
            )
            +
            "/product/"
            +
            random.choice(
                [
                    "A",
                    "B",
                    "C"
                ]
            )
    )


def generate_content():
    r = random.random()

    if r < 0.3:
        return random_number()
    elif r < 0.7:
        return random_text()
    else:
        return random_url()


def generate_qr(data, ecc):
    # 大部分自动版本
    if random.random() < 0.8:
        version = None
    else:
        version = random.randint(1, 8)
    qr = qrcode.QRCode(
        version=version,
        error_correction=ecc,
        box_size=10,
        border=4
    )
    qr.add_data(data)

    qr.make(fit=True)

    img = qr.make_image(
        fill_color="black",
        back_color="white"
    ).convert("RGB")

    return img


def resize_keep_border(img):
    target = random.randint(150, 220)

    img.thumbnail(
        (target, target),
        Image.Resampling.LANCZOS
    )
    canvas = Image.new(
        "RGB",
        (SIZE, SIZE),
        "white"
    )

    x = (SIZE - img.width) // 2
    y = (SIZE - img.height) // 2

    canvas.paste(img, (x, y))

    return canvas


with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "filename",
            "content",
            "ecc"
        ]
    )

    for i in range(NUM_IMAGES):

        content = generate_content()
        ecc = random_ecc()
        qr_img = generate_qr(content, ecc)

        qr_img = resize_keep_border(qr_img)

        filename = f"{i + 1:05d}.png"

        path = os.path.join(SAVE_DIR, filename)
        qr_img.save(path)
        writer.writerow(
            [
                filename,
                content,
                ECC_MAP[ecc]
            ]
        )
        if (i + 1) % 500 == 0:
            print(
                f"{i + 1}/{NUM_IMAGES}"
            )

print("done")
