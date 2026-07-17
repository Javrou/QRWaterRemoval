import torch

from tqdm import tqdm
from torch.cuda.amp import autocast

from loss import compute_loss
from engine.metrics import evaluate_metrics
from utils.visualizer import Visualizer


class Evaluator:

    def __init__(
            self,
            model,
            device,
            visual_dir=None
    ):

        self.model = model
        self.device = device
        self.visualizer = None

        if visual_dir is not None:
            self.visualizer = Visualizer(visual_dir)

    def evaluate(
            self,
            loader,
            mode="pretrain",
            save_visual=False,
            epoch=None
    ):

        self.model.eval()

        loss_sum = 0.0
        metric_sum = {
            "psnr": 0.0,
            "ssim": 0.0,
            "binary_acc": 0.0,
            "zxing": 0.0
        }

        with torch.no_grad():
            for i, (inp, tgt) in enumerate(tqdm(loader, leave=False)):
                inp = inp.to(self.device, non_blocking=True)
                tgt = tgt.to(self.device, non_blocking=True)

                with autocast(device_type="cuda"):

                    pred = self.model(inp).clamp(0, 1)
                    loss = compute_loss(pred, tgt, mode)

                loss_sum += loss.item()
                metrics = evaluate_metrics(pred, tgt)

                for k in metric_sum:
                    metric_sum[k] += metrics[k]

                if save_visual and self.visualizer is not None and i == 0:

                    self.visualizer.save_batch(
                        inp,
                        pred,
                        tgt,
                        prefix=f"epoch{epoch}"
                    )

        n = len(loader)

        return {
            "loss": loss_sum / n,
            "psnr": metric_sum["psnr"] / n,
            "ssim": metric_sum["ssim"] / n,
            "binary_acc": metric_sum["binary_acc"] / n,
            "zxing": metric_sum["zxing"] / n
        }

    @staticmethod
    def batch_zxing(pred):

        return evaluate_metrics(
            pred,
            pred
        )["zxing"]