import os
import csv


class StepLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "epoch",
                    "step",
                    "loss",
                    "lr",
                    "grad_norm",
                    "zxing",
                    "binary_acc"
                ])

    def log(self, epoch, step, loss, lr, grad_norm=None, zxing=None, binary_acc=None):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                step,
                loss,
                lr,
                grad_norm,
                zxing,
                binary_acc
            ])


class EpochLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "lr",
                    "zxing",
                    "psnr",
                    "ssim",
                    "binary_acc",
                    "best_loss",
                    "best_zxing",
                    "best_psnr",
                    "best_ssim"
                ])

    def log(
        self,
        epoch,
        train_loss,
        val_loss,
        lr,
        zxing,
        psnr,
        ssim,
        binary_acc,
        best_loss,
        best_zxing,
        best_psnr,
        best_ssim
    ):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                train_loss,
                val_loss,
                lr,
                zxing,
                psnr,
                ssim,
                binary_acc,
                best_loss,
                best_zxing,
                best_psnr,
                best_ssim
            ])