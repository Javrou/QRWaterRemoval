import time
import torch.optim as optim
from torch.amp import autocast, GradScaler

import zxingcpp

from model.restormer import Restormer
from tool.data_loader import *
from loss import *
from tool.train_logger import *
from tool.checkpoints import save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)
print("Torch:", torch.__version__)

# ======================
# Model
# ======================

model = Restormer(
    inp_channels=3,
    out_channels=3,
    dim=24,
    num_blocks=[2, 2, 2, 3],
    num_refinement_blocks=1,
    heads=[1, 2, 2, 4],
    ffn_expansion_factor=2.0,
    bias=False,
    LayerNorm_type='WithBias'
).to(device)

print(
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
)

# ======================
# 加载预训练 v1 模型
# ======================
checkpoint_path = "checkpoints/best_v1.pth"
checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)
model.load_state_dict(
    checkpoint["model"]
)

print(
    "Loaded pretrained model:",
    checkpoint_path
)

# ======================
# Optimizer
# ======================

optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4
)

scaler = GradScaler("cuda")


# ======================
# Metrics
# ======================

def zxing_rate(batch_tensor):
    success = 0
    total = batch_tensor.shape[0]

    imgs = batch_tensor.detach().cpu()

    for i in range(total):

        img = imgs[i]

        img = img.permute(1, 2, 0).numpy()

        img = (img * 255) \
            .clip(0, 255) \
            .astype("uint8")

        results = zxingcpp.read_barcodes(img)

        if len(results) > 0:
            success += 1

    return success / total


# ======================
# PSNR
# ======================
def calculate_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)

    if mse == 0:
        return 100

    return (
            10 *
            torch.log10(
                1.0 / mse
            )
    ).item()


# ======================
# Validation
# ======================
def validate(model):
    model.eval()

    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_success = 0
    total_num = 0

    with torch.no_grad():
        for inp, tgt in val_loader:
            inp = inp.to(device)
            tgt = tgt.to(device)

            pred = model(inp)
            pred = pred.clamp(0, 1)

            # loss
            loss = compute_loss(pred, tgt)
            total_loss += loss.item()

            # PSNR
            total_psnr += calculate_psnr(
                pred,
                tgt
            )

            # SSIM
            ssim_value = (
                    1 - ssim_loss(pred, tgt)
            )
            total_ssim += ssim_value.item()

            # ZXing
            sr = zxing_rate(pred)

            total_success += (
                    sr * inp.size(0)
            )

            total_num += inp.size(0)

    avg_loss = total_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    avg_ssim = total_ssim / len(val_loader)
    zxing = total_success / total_num

    return (
        avg_loss,
        zxing,
        avg_psnr,
        avg_ssim
    )


# ======================
# Training
# ======================

logger = CSVLogger("logs/finetune_log.csv")

num_epochs = 5
global_step = 0
best_zxing = 0

for epoch in range(num_epochs):

    model.train()
    running_loss = 0
    t_epoch = time.time()

    for i, (inp, tgt) in enumerate(train_loader):
        t0 = time.time()

        inp = inp.to(
            device,
            non_blocking=True
        )

        tgt = tgt.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            pred = model(inp)

            loss = compute_loss(
                pred,
                tgt
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1e9
        ).item()
        scaler.step(
            optimizer
        )
        scaler.update()

        running_loss += loss.item()

        global_step += 1

        if (i + 1) % 10 == 0:
            print(
                f"[E{epoch} "
                f"{i + 1}/{len(train_loader)}] "
                f"loss={loss.item():.4f} "
                f"time={time.time() - t0:.3f}s "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"grad={grad_norm:.2f}"
            )

    # ======================
    # Epoch validation
    # ======================

    train_loss = (
            running_loss /
            len(train_loader)
    )

    val_loss, val_zxing, val_psnr, val_ssim = validate(model)

    print("\n======================")
    print(f"Epoch {epoch} finished")
    print("Time:", time.time() - t_epoch)
    print(f"Train Loss: {train_loss:.6f}")
    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation ZXing Rate: "f"{val_zxing:.4f}")
    print(f"Validation PSNR: "f"{val_psnr:.4f}")
    print(f"Validation SSIM: "f"{val_ssim:.4f}")
    print("======================\n")

    logger.log(
        epoch,
        global_step,
        val_loss,
        optimizer.param_groups[0]['lr'],
        val_zxing,
        0
    )

    # ======================
    # checkpoint
    # ======================

    save_checkpoint(
        "checkpoints/real_latest.pth",
        model,
        optimizer,
        None,
        epoch,
        global_step,
        val_loss
    )

    # 根据ZXing保存最佳模型
    if val_zxing > best_zxing:
        best_zxing = val_zxing

        save_checkpoint(
            "checkpoints/real_best.pth",
            model,
            optimizer,
            None,
            epoch,
            global_step,
            val_loss
        )

print("Training finished")
