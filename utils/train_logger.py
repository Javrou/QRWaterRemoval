import csv
import os
import time


class StepLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existed = os.path.exists(path)
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not existed:
            self._writer.writerow([
                "timestamp", "epoch", "step",
                "loss", "lr", "grad_norm", "zxing"
            ])
            self._file.flush()

    def log(self, epoch, step, loss, lr, grad_norm, zxing):
        self._writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            epoch, step,
            f"{loss:.6f}" if loss is not None else "",
            f"{lr:.8f}" if lr is not None else "",
            f"{grad_norm:.4f}" if grad_norm is not None else "",
            f"{zxing:.4f}" if zxing is not None else ""
        ])
        self._file.flush()

    @staticmethod
    def print(epoch, batch, total_batch, loss, lr, grad_norm, elapsed):
        print(
            f"[Epoch {epoch:3d}] "
            f"[{batch:4d}/{total_batch}] "
            f"Loss: {loss:.6f} | "
            f"LR: {lr:.2e} | "
            f"Grad: {grad_norm:.2f} | "
            f"Time: {elapsed:.2f}s"
        )

    @staticmethod
    def print_zxing(batch, zxing):
        print(f"  [Batch {batch:4d}] ZXing SR: {zxing:.4f}")

    def close(self):
        self._file.close()


class EpochLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existed = os.path.exists(path)
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if not existed:
            self._writer.writerow([
                "timestamp", "epoch",
                "train_loss", "val_loss", "lr",
                "zxing", "psnr", "ssim", "binary_acc",
                "best_loss", "best_zxing", "best_psnr", "best_ssim"
            ])
            self._file.flush()

    def log(self, epoch, train_loss, val_loss, lr,
            zxing, psnr, ssim, binary_acc,
            best_loss, best_zxing, best_psnr, best_ssim):
        self._writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            epoch,
            f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.8f}",
            f"{zxing:.4f}", f"{psnr:.4f}", f"{ssim:.4f}", f"{binary_acc:.4f}",
            f"{best_loss:.6f}", f"{best_zxing:.4f}",
            f"{best_psnr:.4f}", f"{best_ssim:.4f}"
        ])
        self._file.flush()

    @staticmethod
    def print(epoch, epoch_time, train_loss, metrics):
        print(
            f"\n===== Epoch {epoch:3d} | Time: {epoch_time:.1f}s ====="
        )
        print(
            f"Train Loss: {train_loss:.6f}"
        )
        print(
            f"Val Loss  : {metrics['loss']:.6f} | "
            f"ZXing: {metrics['zxing']:.4f} | "
            f"PSNR: {metrics['psnr']:.4f} | "
            f"SSIM: {metrics['ssim']:.4f} | "
            f"BinAcc: {metrics['binary_acc']:.4f}"
        )

    def close(self):
        self._file.close()
