"""
val & test input数据生成
"""
import cv2
import random
import numpy as np
from pathlib import Path

from tool.degradation import degrade_qr


def generate_degraded_dataset(src_dir, save_dir, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    src_dir = Path(src_dir)
    save_dir = Path(save_dir)

    input_dir = save_dir / "input"
    target_dir = save_dir / "target"

    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        src_dir.glob("*.png")
    )

    print("Total:", len(files))

    for idx, file in enumerate(files):

        img = cv2.imread(
            str(file),
            cv2.IMREAD_COLOR
        )

        if img is None:
            continue

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        cv2.imwrite(
            str(target_dir / file.name),
            cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR
            )
        )

        degraded = degrade_qr(img)

        cv2.imwrite(
            str(input_dir / file.name),
            cv2.cvtColor(
                degraded,
                cv2.COLOR_RGB2BGR
            )
        )

        if (idx + 1) % 100 == 0:
            print(
                f"{idx + 1}/{len(files)}"
            )

    print("Finished:", save_dir)


if __name__ == "__main__":
    # validation

    generate_degraded_dataset(
        "../raw_data/synthetic/val/target",
        "raw_data/synthetic/val",
        seed=100
    )

    # test

    generate_degraded_dataset(
        "../raw_data/synthetic/test/target",
        "raw_data/synthetic/test",
        seed=200
    )
