import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

l1_loss = nn.L1Loss()
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)


# ======================
# Binary Loss
# ======================
def binary_loss(pred, gt):
    with torch.amp.autocast(device_type="cuda", enabled=False):

        pred = pred.float()
        gt = gt.float()
        pred_gray = torch.clamp(pred, 1e-6, 1-1e-6)

        return F.binary_cross_entropy(
            pred_gray,
            gt
        )


# ======================
# Edge Loss
# ======================
def edge_loss(pred, gt):
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)

    def edge(img):
        gx = F.conv2d(
            img,
            sobel_x,
            padding=1
        )
        gy = F.conv2d(
            img,
            sobel_y,
            padding=1
        )
        return torch.sqrt(
            gx ** 2 +
            gy ** 2 +
            1e-6
        )

    return F.l1_loss(edge(pred), edge(gt))


# ======================
# ZXing Proxy Loss
# ======================
def zxing_proxy_loss(pred, tau=0.1):
    soft_bin = torch.sigmoid((pred - 0.5) / tau)

    loss_soft = F.l1_loss(soft_bin, pred)
    entropy_penalty = torch.mean(soft_bin * (1 - soft_bin))

    return loss_soft + 0.5 * entropy_penalty


# ======================
# Total Loss
# ======================
def compute_loss(pred, gt, mode="finetune"):
    l1 = l1_loss(pred, gt)
    ssim = ssim_loss(pred, gt)

    if mode == "pretrain":
        loss = (
            1.0 * l1 +
            0.05 * ssim +
            0.30 * edge_loss(pred, gt) +
            0.25 * binary_loss(pred, gt) +
            0.10 * zxing_proxy_loss(pred)
        )
    elif mode == "finetune":
        loss = (
            0.8 * l1 +
            0.05 * ssim +
            0.35 * edge_loss(pred, gt) +
            0.30 * binary_loss(pred, gt) +
            0.15 * zxing_proxy_loss(pred)
        )

    return loss
