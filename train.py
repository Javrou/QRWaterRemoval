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

# 轻量restormer
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

print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

# optimizer with AMP
optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)
scaler = GradScaler("cuda")


# ZXing eval
def zxing_rate(batch_tensor):
    success = 0
    total = batch_tensor.shape[0]
    imgs = batch_tensor.detach().cpu()

    for i in range(total):
        img = imgs[i]
        # CHW -> HWC
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype("uint8")
        results = zxingcpp.read_barcodes(img)
        if len(results) > 0:
            success += 1

    return success / total


# ======================
# Training
# ======================
logger = CSVLogger('logs/train_log.csv')

best_loss = 1e9
global_step = 0

num_epochs = 5

running_loss = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    t_epoch = time.time()

    for i, (inp, tgt) in enumerate(train_loader):
        t0 = time.time()

        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # forward
        with autocast(device_type="cuda"):
            pred = model(inp)
            loss = compute_loss(pred, tgt)

        # backward
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        global_step += 1

        dt = time.time() - t0

        # log
        if (i + 1) % 10 == 0:
            print(
                f"[E{epoch} {i + 1}/{len(train_loader)}] "
                f"loss={loss.item():.4f} "
                f"time={dt:.3f}s "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"grad_norm={total_norm:.2f}"
            )
            logger.log(
                epoch=epoch,
                step=global_step,
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr'],
                zxing_rate=None,
                grad_norm=total_norm
            )

        # zxing eval (每 50 step)
        if (i + 1) % 50 == 0:
            with torch.no_grad():
                pred_vis = pred.clamp(0, 1)
                sr = zxing_rate(pred_vis)
            print(f"[ZXing @ step {i + 1}] {sr:.4f}")

            logger.log(
                epoch=epoch,
                step=global_step,
                loss=loss.item(),
                lr=optimizer.param_groups[0]['lr'],
                zxing_rate=sr,
                grad_norm=total_norm
            )

    # ======================
    # Epoch summary
    # ======================
    avg_loss = running_loss / len(train_loader)

    print("\nEpoch finished")
    print("Epoch time:", time.time() - t_epoch)
    print("Avg Loss:", avg_loss)

    # ======================
    # Validation
    # ======================
    model.eval()

    with torch.no_grad():
        inp, tgt = next(iter(val_loader))
        inp = inp.to(device)

        pred = model(inp).clamp(0, 1)

        sr = zxing_rate(pred)

    print("Validation ZXing Rate:", sr)

    # ======================
    # checkpoint
    # ======================
    save_checkpoint(
        "checkpoints/latest.pth",
        model, optimizer, None,
        epoch, global_step, best_loss
    )
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(
            "checkpoints/best.pth",
            model, optimizer, None,
            epoch, global_step, best_loss
        )