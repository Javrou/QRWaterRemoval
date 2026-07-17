import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


# ======================
# Basic Loss
# ======================
def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def ssim_loss(pred, target):
    return 1 - ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True
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
# ROI L1 — 受损区域加权 L1
# ======================
def build_roi_mask(gt):
    return (gt < 0.8).float()


def roi_l1_loss(pred, gt):
    mask = build_roi_mask(gt)
    diff = torch.abs(pred - gt)
    loss = (diff * mask).sum()
    loss /= (mask.sum() + 1e-6)

    return loss


# ======================
# ZXing Proxy Loss
# ======================
def zxing_proxy_loss(pred, tau=0.1):
    """将 ZXing 解码器的硬阈值行为用可微的 sigmoid 近似。
    - loss_soft: 软二值化结果 (≈0/1) 与原始预测之间的 L1，
      推动输出远离中间灰度
    - entropy_penalty: u*(1-u) 形式的双峰正则化，
      基于 Ginzburg-Landau / Mumford-Shah 思想，
      惩罚预测值在 0.3~0.7 之间的模糊像素
    tau 控制 sigmoid 的陡峭程度：越小越接近硬二值化，梯度越稀疏。
    0.1 在可微性和二值化效果之间取得平衡。
    """
    soft_bin = torch.sigmoid((pred - 0.5) / tau)

    loss_soft = F.l1_loss(soft_bin, pred)
    entropy_penalty = torch.mean(soft_bin * (1 - soft_bin))

    return loss_soft + 0.5 * entropy_penalty


# ======================
# Binary Loss
# ======================
def binary_loss(pred, gt):

    with torch.amp.autocast(device_type="cuda", enabled=False):
        pred = pred.float()
        gt = gt.float()

        pred = pred.clamp(1e-6, 1-1e-6)
        gt_bin = (gt > 0.5).float()

        return F.binary_cross_entropy(pred,gt_bin)


# ======================
# 频域幅度损失 (Focal Frequency Loss)
# ======================
def fft_loss(pred, target):
    """频域幅度损失 — 保留二维码的周期性结构

    二维码的定位图案（三层同心正方形）、校正图案和模块网格
    具有强烈且特定的空间频率特征。水渍损伤通常引入低频/中频伪影
    （不均匀暗化），频域损失可以有效抑制这些伪影并恢复原始的
    周期结构。

    对数幅度压缩使不同频段（低频 DC 到高频边缘）的梯度贡献更均衡，
    避免 DC 分量主导损失。
    """
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")

    pred_mag = torch.log(torch.abs(pred_fft) + 1e-8)
    target_mag = torch.log(torch.abs(target_fft) + 1e-8)
    return F.l1_loss(pred_mag, target_mag)


# ======================
# Soft Dice Loss
# ======================
def soft_dice_loss(pred, target, smooth=1.0):
    pred = 1 - pred
    target = 1 - target
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return 1-dice


# ======================
# Total Loss
# ======================
def compute_loss(pred, gt, mode="pretrain"):
    """两阶段权重策略:
    - 预训练: 合成数据退化规律已知，侧重重建能力。
      L1/SSIM 主导，FFT/Dice/ZXing 弱辅助建立对二维码结构的初步感知。
    - 微调: 真实数据的退化分布偏移更大。
      L1 降权避免过度平滑，FFT 和 ZXing 升权强调频域结构
      和二值化质量——因为最终评判指标是 ZXing 解码率而非 PSNR。
    """
    l1 = l1_loss(pred, gt)
    ssim_v = ssim_loss(pred, gt)
    roi = roi_l1_loss(pred, gt)
    edge = edge_loss(pred, gt)
    fft = fft_loss(pred, gt)
    dice = soft_dice_loss(pred, gt)
    binary_v = binary_loss(pred, gt)
    zxing = zxing_proxy_loss(pred)

    if mode == "pretrain":
        return (
            0.45 * l1
            + 0.15 * ssim_v
            + 0.20 * roi
            + 0.10 * edge
            + 0.10 * fft
            + 0.10 * dice
            + 0.05 * binary_v
            + 0.05 * zxing
        )
    else:   # finetune
        return (
            0.45 * l1
            + 0.15 * ssim_v
            + 0.20 * roi
            + 0.10 * edge
            + 0.15 * fft
            + 0.10 * dice
            + 0.05 * binary_v
            + 0.05 * zxing
        )
