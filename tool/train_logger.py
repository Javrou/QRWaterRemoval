"""
日志
"""
import os
import csv


class CSVLogger:

    def __init__(self, path):
        self.path = path

        os.makedirs(os.path.dirname(path), exist_ok=True)

        file_exists = os.path.exists(path)

        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "epoch",
                    "step",
                    "loss",
                    "lr",
                    "zxing_rate",
                    "grad_norm"
                ])

    def log(self, epoch, step, loss, lr, zxing_rate, grad_norm):
        with open(self.path, 'a', newline='') as f:
            writer = csv.writer(f)

            writer.writerow([
                epoch,
                step,
                float(loss),
                float(lr),
                float(zxing_rate) if zxing_rate is not None else -1,
                float(grad_norm)
            ])
