"""
数据集重命名
"""
import os
import re

dir_path = r"real_data/raw_target"

# 只取 jpg/png
files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.png'))]

files.sort()

print(f"找到 {len(files)} 张图片")


def extract_time(name):
    # IMG_20260706_102302
    m = re.search(r'IMG_(\d{8})_(\d{6})', name)
    if m:
        return m.group(1) + m.group(2)
    return name


files.sort(key=extract_time)

start_index = 1

for i, fname in enumerate(files, start=start_index):
    ext = os.path.splitext(fname)[1]

    new_name = f"{i:04d}{ext}"

    old_path = os.path.join(dir_path, fname)
    new_path = os.path.join(dir_path, new_name)

    os.rename(old_path, new_path)

    print(f"{fname}  ->  {new_name}")

print("重命名完成")
