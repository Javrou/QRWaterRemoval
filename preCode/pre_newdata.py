import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np


def detect_qr(img):
    detector = cv2.QRCodeDetector()

    ok, points = detector.detect(img)

    if not ok or points is None:
        return None

    return points.reshape(4, 2)


def calc_crop_size(
        pts,
        scale=2.4,
        min_size=2400,
        max_size=3000
):

    w1 = np.linalg.norm(pts[0] - pts[1])
    w2 = np.linalg.norm(pts[2] - pts[3])

    h1 = np.linalg.norm(pts[1] - pts[2])
    h2 = np.linalg.norm(pts[3] - pts[0])

    qr_size = max(w1, w2, h1, h2)

    crop = int(qr_size * scale)

    crop = np.clip(
        crop,
        min_size,
        max_size
    )

    return crop


def crop_by_center(img, center, crop_size):
    h, w = img.shape[:2]

    cx, cy = center
    half = crop_size // 2

    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w
    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)

    crop = img[y1:y2, x1:x2]

    crop = cv2.resize(
        crop,
        (crop_size, crop_size),
        interpolation=cv2.INTER_AREA
    )

    return crop


def process_pair(
        input_path,
        target_path,
        save_input,
        save_target,
        out_size=256
):

    input_img = cv2.imread(str(input_path))
    target_img = cv2.imread(str(target_path))

    if input_img is None or target_img is None:
        return False

    pts = detect_qr(target_img)

    if pts is None:
        return False

    center = pts.mean(axis=0)

    crop_size = calc_crop_size(
        pts,
        scale=2.4,
        min_size=2400,
        max_size=3000
    )

    crop_input = crop_by_center(
        input_img,
        center,
        crop_size
    )

    crop_target = crop_by_center(
        target_img,
        center,
        crop_size
    )

    crop_input = cv2.resize(
        crop_input,
        (out_size, out_size),
        interpolation=cv2.INTER_AREA
    )

    crop_target = cv2.resize(
        crop_target,
        (out_size, out_size),
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite(str(save_input), crop_input)
    cv2.imwrite(str(save_target), crop_target)

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=Path, default=Path("../new_real_data/input"))
    parser.add_argument("--target_dir", type=Path, default=Path("../new_real_data/target"))
    parser.add_argument("--output_dir", type=Path, default=Path("../new_division_data"))
    parser.add_argument("--start_index", type=int, default=1120)
    parser.add_argument("--end_index", type=int, default=1678)

    parser.add_argument("--crop_size", type=int, default=3400)
    parser.add_argument("--size", type=int, default=256)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)

    output_dir = Path(args.output_dir)
    output_input = output_dir / "input"
    output_target = output_dir / "target"

    output_input.mkdir(parents=True, exist_ok=True)
    output_target.mkdir(parents=True, exist_ok=True)

    failed = []

    files = []

    for p in sorted(target_dir.glob("*.jpg")):
        idx = int(p.stem)
        if args.start_index <= idx <= args.end_index:
            files.append(p)

    for target_path in tqdm(files):

        name = target_path.name

        ok = process_pair(
            input_dir / name,
            target_path,
            output_input / name,
            output_target / name,
            out_size=256
        )

        if not ok:
            failed.append(name)

    failed_path = output_dir / "failed.txt"

    with open(failed_path, "a", encoding="utf-8") as f:
        for x in failed:
            f.write(x + "\n")

    print("=" * 40)
    print("Finished")
    print("Total :", len(files))
    print("Failed:", len(failed))
    print("=" * 40)


if __name__ == "__main__":
    main()
