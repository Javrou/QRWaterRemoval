import os

# 数据集路径
root_dir = r"../new_division_data"
input_dir = os.path.join(root_dir, "input")
target_dir = os.path.join(root_dir, "target")

# 支持的图片格式
IMG_EXTS = {".png", ".jpg"}

# 获取两个文件夹中的文件
input_files = {
    f for f in os.listdir(input_dir)
    if os.path.splitext(f)[1].lower() in IMG_EXTS
}

target_files = {
    f for f in os.listdir(target_dir)
    if os.path.splitext(f)[1].lower() in IMG_EXTS
}

# 只保留两边都有的文件
common_files = sorted(input_files & target_files)

print(f"找到 {len(common_files)} 对图片")

temp_pairs = []

for idx, filename in enumerate(common_files):
    ext = os.path.splitext(filename)[1].lower()

    input_old = os.path.join(input_dir, filename)
    target_old = os.path.join(target_dir, filename)

    input_tmp = os.path.join(input_dir, f"__tmp__{idx}{ext}")
    target_tmp = os.path.join(target_dir, f"__tmp__{idx}{ext}")

    os.rename(input_old, input_tmp)
    os.rename(target_old, target_tmp)

    temp_pairs.append((input_tmp, target_tmp, ext))

for idx, (input_tmp, target_tmp, ext) in enumerate(temp_pairs, start=1):
    new_name = f"{idx:04d}{ext}"

    input_new = os.path.join(input_dir, new_name)
    target_new = os.path.join(target_dir, new_name)

    os.rename(input_tmp, input_new)
    os.rename(target_tmp, target_new)

print("重命名完成！")
print(f"共处理 {len(temp_pairs)} 对图片")