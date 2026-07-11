import os
import cv2
import numpy as np
import torch


def tensor2img(x):
    x = x.detach().cpu().clamp(0, 1)
    x = x[0].numpy()
    x = (x*225).astype(np.uint8)
    return x


def save_visual_results(
        model,
        loader,
        device,
        epoch,
        save_dir="visual",
        indices=(0, 10, 20, 50)
):
    model.eval()

    epoch_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)

    with torch.no_grad():
        sample_id = 0
        for inp, tgt in loader:

            inp = inp.to(device)
            pred = model(inp).clamp(0, 1)

            bs = inp.size(0)

            for b in range(bs):

                if sample_id in indices:
                    input_img = tensor2img(inp[b])
                    pred_img = tensor2img(pred[b])
                    target_img = tensor2img(tgt[b])

                    result = np.concatenate(
                        [
                            input_img,
                            pred_img,
                            target_img
                        ],
                        axis=1
                    )
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

                    cv2.putText(
                        result,
                        "Input",
                        (20, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        result,
                        "Prediction",
                        (280, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        result,
                        "Target",
                        (560, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.imwrite(
                        os.path.join(epoch_dir, f"sample_{sample_id:04d}.png"),
                        result
                    )

                sample_id += 1
