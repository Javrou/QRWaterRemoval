import os
import torch

from pretrain import pretrain
from finetune import finetune


def main():
    print("====================")
    print("Stage 1: Pretrain")
    print("====================")
    pretrain_path = pretrain()
    if not os.path.exists(pretrain_path):
        raise RuntimeError(
            "Pretrain checkpoint not found"
        )

    print("====================")
    print("Stage 2: Fine-tune")
    print("====================")
    finetune(checkpoint_path=pretrain_path)
    print("====================")
    print("ALL TRAINING FINISHED")
    print("====================")


if __name__ == "__main__":
    main()
