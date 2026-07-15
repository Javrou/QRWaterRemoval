import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def detect_qr(img):
    detector = cv2.QRCodeDetector()
    ok, points = detector.detect(img)

    if not ok or points is None:
        return None

    return points.reshape(4, 2)


def calc_crop_size(
        pts,
        scale=2.2,
        min_size=1800,
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


def crop_by_center(
        img,
        center,
        crop_size
):
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


def split_2x2(img):
    h, w = img.shape[:2]

    cx = w // 2
    cy = h // 2

    return [
        img[:cy, :cx],
        img[:cy, cx:],
        img[cy:, :cx],
        img[cy:, cx:]
    ]


def process_pair(
        input_path,
        target_path,
        save_input,
        save_target,
        start_index,
        out_size=256
):

    input_img = cv2.imread(str(input_path))
    target_img = cv2.imread(str(target_path))

    if input_img is None or target_img is None:
        return start_index, ["Image Read Failed"]

    input_blocks = split_2x2(input_img)
    target_blocks = split_2x2(target_img)

    failed = []
    index = start_index

    for block_id in range(4):

        input_crop = input_blocks[block_id]
        target_crop = target_blocks[block_id]

        pts = detect_qr(target_crop)

        if pts is None:
            failed.append(f"{target_path.name} block{block_id+1} -> {index:04d}.png")

            crop_input = cv2.resize(
                input_crop,
                (out_size, out_size),
                interpolation=cv2.INTER_AREA
            )

            crop_target = cv2.resize(
                target_crop,
                (out_size, out_size),
                interpolation=cv2.INTER_AREA
            )

            save_name = f"{index:04d}.png"

            cv2.imwrite(
                str(save_input / save_name),
                crop_input
            )

            cv2.imwrite(
                str(save_target / save_name),
                crop_target
            )

            index += 1

            continue

        center = pts.mean(axis=0)

        crop_size = calc_crop_size(
            pts,
            scale=2.2,
            min_size=900,
            max_size=1700
        )

        crop_input = crop_by_center(
            input_crop,
            center,
            crop_size
        )

        crop_target = crop_by_center(
            target_crop,
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

        save_name = f"{index:04d}.jpg"

        cv2.imwrite(str(save_input / save_name), crop_input)
        cv2.imwrite(str(save_target / save_name), crop_target)

        index += 1

    return index, failed


def main():
    input_dir = Path("../new_4real_data/input")
    target_dir = Path("../new_4real_data/target")

    output_dir = Path("../new_division_data")

    output_input = output_dir / "input"
    output_target = output_dir / "target"

    output_input.mkdir(parents=True, exist_ok=True)
    output_target.mkdir(parents=True, exist_ok=True)

    start_index = 4055

    failed_path = output_dir / "failed.txt"

    files = sorted(target_dir.glob("*.jpg"))

    failed_all = []

    for target_path in tqdm(files):
        input_path = input_dir / target_path.name

        start_index, failed = process_pair(
            input_path,
            target_path,
            output_input,
            output_target,
            start_index,
            out_size=256
        )

        failed_all.extend(failed)
    old_failed = set()

    if failed_path.exists():
        with open(failed_path, "r") as f:
            old_failed = {
                line.strip()
                for line in f
                if line.strip()
            }

    old_failed.update(failed_all)

    with open(failed_path, "w") as f:
        for x in sorted(old_failed):
            f.write(x + "\n")

    print("=" * 40)
    print("Finished")
    print("Images      :", len(files))
    print("Output QR   :", start_index - 1)
    print("Failed QR   :", len(failed_all))
    print("Next Index  :", start_index)
    print("=" * 40)


if __name__ == "__main__":
    main()
