"""
zxing测试扫描成功率
"""
import cv2
import zxingcpp
from pathlib import Path
import time

# ==========================
# 配置
# ==========================

IMAGE_DIR = "../raw_data/7k_real_dataset/png_target"

# 支持格式
EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp"
]


# ==========================
# ZXing检测函数
# ==========================

def zxing_decode(img):
    # BGR -> RGB
    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    results = zxingcpp.read_barcodes(img_rgb)

    if len(results) > 0:
        return True, results[0].text

    return False, None


def main():
    image_paths = []

    for ext in EXTENSIONS:
        image_paths.extend(
            Path(IMAGE_DIR).glob("*" + ext)
        )

    image_paths = sorted(
        image_paths,
        key=lambda x: x.name
    )

    total = len(image_paths)

    if total == 0:
        print("没有找到图片")
        return

    print("=" * 50)
    print(f"测试图片数量: {total}")
    print("=" * 50)

    success = 0

    t0 = time.time()

    for idx, path in enumerate(image_paths):

        img = cv2.imread(
            str(path)
        )

        if img is None:
            print(
                f"[{idx + 1}/{total}] "
                f"{path.name} 读取失败"
            )
            continue

        ok, text = zxing_decode(img)
        if ok:
            success += 1

    elapsed = time.time() - t0

    rate = success / total

    print("\n")
    print("=" * 50)
    print("ZXing Test Result")
    print("=" * 50)

    print(
        f"Total images : {total}"
    )

    print(
        f"Success      : {success}"
    )

    print(
        f"Fail         : {total - success}"
    )

    print(
        f"Success Rate : {rate:.4f}"
    )

    print(
        f"Success Rate : {rate * 100:.2f}%"
    )

    print(
        f"Time         : {elapsed:.2f}s"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()
