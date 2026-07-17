import math
import numpy as np

from skimage.metrics import structural_similarity
import zxingcpp


# ==========================
# PSNR
# ==========================
def calc_psnr(pred, target):

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    mse = np.mean((pred - target) ** 2)

    if mse < 1e-10:
        return 100.0

    return 20 * math.log10(1.0 / math.sqrt(mse))


# ==========================
# SSIM
# ==========================
def calc_ssim(pred, target):

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    total = 0
    for p, t in zip(pred, target):
        p = np.squeeze(p)
        t = np.squeeze(t)

        total += structural_similarity(
            p,
            t,
            data_range=1.0
        )

    return total / len(pred)


# ==========================
# Binary Accuracy
# ==========================
def calc_binary_acc(
        pred,
        target,
        threshold=0.5
):

    pred = (pred >= threshold).float()

    return (
        pred.eq(target)
        .float()
        .mean()
        .item()
    )


# ==========================
# ZXing
# ==========================
def zxing_rate(
        imgs,
        threshold=0.5
):

    imgs = imgs.detach().cpu()
    success = 0
    for img in imgs:
        img = img.squeeze().numpy()
        img = (
            (img > threshold)
            .astype(np.uint8)
            * 255
        )
        result = zxingcpp.read_barcodes(img)

        if len(result):
            success += 1

    return success / len(imgs)