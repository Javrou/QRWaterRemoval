import torch
import torch.nn.functional as F
import zxingcpp

from loss import *


def zxing_rate(batch_tensor):
    success = 0
    total = batch_tensor.shape[0]
    imgs = batch_tensor.detach().cpu()

    for i in range(total):
        img = imgs[i]
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        result = zxingcpp.read_barcodes(img)
        if len(result) > 0:
            success += 1

    return success / total


def calculate_psnr(pred, gt):
    mse = F.mse_loss(pred, gt)
    if mse == 0:
        return 100

    return (10 * torch.log10(1.0 / mse)).item()


def binary_accuracy(pred, gt):
    pred_gray = (
            0.299 * pred[:, 0:1] +
            0.587 * pred[:, 1:2] +
            0.114 * pred[:, 2:3]
    )
    gt_gray = (
            0.299 * gt[:, 0:1] +
            0.587 * gt[:, 1:2] +
            0.114 * gt[:, 2:3]
    )

    pred_bin = (pred_gray > 0.5)
    gt_bin = (gt_gray > 0.5)
    acc = (pred_bin == gt_bin).float().mean()

    return acc.item()


def validate(model, loader, device, mode="pretrain"):
    model.eval()

    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_success = 0
    total_num = 0
    total_binary_acc = 0

    with torch.no_grad():
        for inp, tgt in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)
            pred = model(inp)
            pred = pred.clamp(0, 1)
            # loss
            loss = compute_loss(pred, tgt, mode=mode)
            total_loss += loss.item()
            # PSNR
            total_psnr += calculate_psnr(pred, tgt)
            # SSIM
            ssim_value = (1 - ssim_loss(pred, tgt))
            total_ssim += ssim_value.item()
            # ZXing
            success = 0
            imgs = pred.cpu()
            for i in range(imgs.shape[0]):
                img = imgs[i]
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype("uint8")
                result = zxingcpp.read_barcodes(img)
                if len(result) > 0:
                    success += 1
            total_success += success
            total_num += inp.size(0)
            # Binary Accuracy
            total_binary_acc += binary_accuracy(pred, tgt)
    return {
        "loss": total_loss / len(loader),
        "zxing": total_success / total_num,
        "psnr": total_psnr / len(loader),
        "ssim": total_ssim / len(loader),
        "binary_acc": total_binary_acc / len(loader)
    }
