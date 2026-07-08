"""
数据集划分
"""
from pathlib import Path
import shutil
import random

random.seed(42)

root = Path("../raw_data/real_data")

input_dir = root / "input"
target_dir = root / "target"

names = sorted([p.name for p in input_dir.glob("*")])

random.shuffle(names)

n = len(names)

train = names[:int(0.8*n)]
val   = names[int(0.8*n):int(0.9*n)]
test  = names[int(0.9*n):]

for split, files in zip(
    ["train","val","test"],
    [train,val,test]
):

    (root/split/"input").mkdir(parents=True,exist_ok=True)
    (root/split/"target").mkdir(parents=True,exist_ok=True)

    for f in files:

        shutil.copy(
            input_dir/f,
            root/split/"input"/f
        )

        shutil.copy(
            target_dir/f,
            root/split/"target"/f
        )

print("Done")