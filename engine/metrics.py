from utils.metrics import (
    calc_psnr,
    calc_ssim,
    calc_binary_acc,
    zxing_rate
)


def evaluate_metrics(
        pred,
        target
):

    return {
        "psnr": calc_psnr(pred, target),
        "ssim": calc_ssim(pred, target),
        "binary_acc": calc_binary_acc(pred, target),
        "zxing": zxing_rate(pred)
    }