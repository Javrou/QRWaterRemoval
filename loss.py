import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

l1_loss = nn.L1Loss()
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)


# ======================
# Binary Loss
# ======================
def binary_loss(pred, gt):
    pred_gray = (
            0.299 * pred[:, 0:1]
            +
            0.587 * pred[:, 1:2]
            +
            0.114 * pred[:, 2:3]
    )

    gt_gray = (
            0.299 * gt[:, 0:1]
            +
            0.587 * gt[:, 1:2]
            +
            0.114 * gt[:, 2:3]
    )

    pred_binary = torch.sigmoid(
        (pred_gray - 0.5) * 10
    )

    gt_binary = (gt_gray > 0.5).float()

    return F.binary_cross_entropy(
        pred_binary,
        gt_binary
    )


# ======================
# RGB -> Gray
# ======================
def rgb2gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


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
        gray = (
                0.299 * img[:, 0:1]
                +
                0.587 * img[:, 1:2]
                +
                0.114 * img[:, 2:3]
        )

        gx = F.conv2d(
            gray,
            sobel_x,
            padding=1
        )

        gy = F.conv2d(
            gray,
            sobel_y,
            padding=1
        )

        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

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
def compute_loss(pred, gt):
    l1 = l1_loss(pred, gt)
    ssim = ssim_loss(pred, gt)

    loss = (
            1.0 * l1 +
            0.10 * ssim +
            0.25 * edge_loss(pred, gt) +
            0.10 * binary_loss(pred, gt) +
            0.10 * zxing_proxy_loss(pred)
    )

    return loss
