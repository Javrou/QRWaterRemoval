from pathlib import Path
import shutil

source_dir = Path(r"../new_4real_data/raw")
output_dir = Path(r"../new_4real_data")

start_index = 665

extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


input_dir = output_dir / "input"
target_dir = output_dir / "target"

input_dir.mkdir(parents=True, exist_ok=True)
target_dir.mkdir(parents=True, exist_ok=True)

files = sorted([
    f for f in source_dir.iterdir()
    if f.suffix in extensions
])

if len(files) % 2 != 0:
    raise RuntimeError("图片数量不是偶数，请检查是否缺少 input 或 target。")

# ======================
# 划分
# ======================
idx = start_index

for i in range(0, len(files), 2):

    input_img = files[i]
    target_img = files[i + 1]

    name = f"{idx:04d}{input_img.suffix.lower()}"

    shutil.copy2(
        input_img,
        input_dir / name
    )

    shutil.copy2(
        target_img,
        target_dir / name
    )

    print(
        f"{input_img.name} -> input/{name}"
    )
    print(
        f"{target_img.name} -> target/{name}"
    )

    idx += 1

print("=" * 40)
print(f"完成，共处理 {(idx - start_index)} 组图片。")
print("=" * 40)