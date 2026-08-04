import torch
from tqdm import tqdm

from builders.model import build_model

# ============================
# Config
# ============================
from configs.pretrain import PretrainConfig
from datasets.builder import build_pretrain_loader
from engine.metrics import evaluate_metrics

cfg = PretrainConfig()

# ============================
# Device
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================
# Load model
# ============================

def load_model(model, path):
    ckpt = torch.load(
        path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        ckpt["model"],
        strict=True
    )

    print("=" * 40)
    print("Loaded checkpoint:")
    print(path)

    if "metrics" in ckpt:
        print(
            "Best ZXing:",
            ckpt["metrics"].get(
                "best_zxing",
                None
            )
        )

    print("=" * 40)


# ============================
# Test
# ============================

@torch.no_grad()
def test(model, loader):
    model.eval()

    total = {
        "psnr": 0,
        "ssim": 0,
        "binary_acc": 0,
        "zxing": 0
    }

    count = 0

    for inp, tgt in tqdm(loader):

        inp = inp.to(
            device,
            non_blocking=True
        )

        tgt = tgt.to(
            device,
            non_blocking=True
        )

        with torch.amp.autocast(
                device_type="cuda"
        ):

            pred = model(inp)

            pred = pred.clamp(
                0, 1
            )

        metrics = evaluate_metrics(
            pred,
            tgt
        )

        for k in total:
            total[k] += metrics[k]

        count += 1

    print("\n========== Test Result ==========")

    print(
        f"ZXing      : {total['zxing'] / count:.4f}"
    )

    print(
        f"PSNR       : {total['psnr'] / count:.4f}"
    )

    print(
        f"SSIM       : {total['ssim'] / count:.4f}"
    )

    print(
        f"Binary Acc : {total['binary_acc'] / count:.4f}"
    )

    print("=================================")


# ============================
# Main
# ============================
if __name__ == "__main__":
    # build model
    model = build_model(cfg, device)

    model.to(device)

    # checkpoint
    ckpt_path = (
            cfg.ckpt_dir
            +
            "/best_zxing.pth"
    )

    load_model(
        model,
        ckpt_path
    )

    _, _, test_loader = build_pretrain_loader(cfg)

    test(
        model,
        test_loader
    )
