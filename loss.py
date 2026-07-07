import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

l1_loss = nn.L1Loss()
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)


# ======================
# Binary Loss
# ======================
def binary_loss(pred):
    return torch.mean(pred * (1 - pred))


# ======================
# RGB -> Gray
# ======================
def rgb2gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


# ======================
# Edge Loss
# ======================
def edge_loss(pred, gt):
    kernel = torch.tensor([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]],
                          dtype=torch.float32,
                          device=pred.device)

    kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,3,3]

    def grad(x):
        x = rgb2gray(x)
        return F.conv2d(x, kernel, padding=1)

    return F.l1_loss(grad(pred), grad(gt))


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
def compute_loss(pred, gt):
    l1 = l1_loss(pred, gt)
    ssim = ssim_loss(pred, gt)

    loss = (
            1.0 * l1 +
            0.15 * ssim +
            0.15 * edge_loss(pred, gt) +
            0.05 * binary_loss(pred) +
            0.08 * zxing_proxy_loss(pred)
    )

    return loss
