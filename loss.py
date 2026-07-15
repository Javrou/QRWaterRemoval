import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

# ======================
# Basic Loss
# ======================
l1_loss = nn.L1Loss()
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=1)


# ======================
# ROI L1
# ======================
def build_roi_mask(gt):
    return (gt < 0.95).float()


def roi_l1_loss(pred, gt):
    mask = build_roi_mask(gt)
    diff = torch.abs(pred - gt)
    loss = (diff * mask).sum()
    loss /= (mask.sum() + 1e-6)

    return loss


# ======================
# Binary Loss
# ======================
def binary_loss(pred, gt):
    with torch.amp.autocast(device_type="cuda", enabled=False):
        pred = pred.float()
        gt = gt.float()

        pred_bin = torch.sigmoid(
            pred * 10
        )
        gt_bin = (gt > 0.5).float()

        return F.binary_cross_entropy(
            pred_bin,
            gt_bin
        )


# ======================
# Edge Loss
# ======================
def edge_loss(pred, gt):
    sobel_x = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        device=pred.device,
        dtype=pred.dtype
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]],
        device=pred.device,
        dtype=pred.dtype
    ).view(1, 1, 3, 3)

    pred_x = F.conv2d(pred, sobel_x, padding=1)
    pred_y = F.conv2d(pred, sobel_y, padding=1)

    gt_x = F.conv2d(gt, sobel_x, padding=1)
    gt_y = F.conv2d(gt, sobel_y, padding=1)

    pred_edge = torch.sqrt(pred_x ** 2 + pred_y ** 2 + 1e-6)
    gt_edge = torch.sqrt(gt_x ** 2 + gt_y ** 2 + 1e-6)

    return F.l1_loss(pred_edge, gt_edge)


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
    roi = roi_l1_loss(pred, gt)
    edge = edge_loss(pred, gt)
    binary = binary_loss(pred, gt)

    if mode == "pretrain":
        loss = (
                1.0 * l1 +
                0.30 * roi +
                0.25 * ssim +
                0.20 * edge +
                0.25 * binary
        )
    elif mode == "finetune":
        loss = (
                0.8 * l1 +
                0.30 * roi +
                0.20 * ssim +
                0.10 * edge +
                0.10 * binary
        )

    return loss
