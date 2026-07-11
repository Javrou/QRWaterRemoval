import os
import cv2
import numpy as np
import torch


def tensor2img(x):
    x = x.detach().cpu().clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    x = (x * 255).astype(np.uint8)
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def save_visual_results(
        model,
        loader,
        device,
        epoch,
        save_dir="visual",
        indices=(0, 10, 20, 50)
):
    model.eval()

    os.makedirs(
        os.path.join(save_dir, f"epoch_{epoch:03d}"),
        exist_ok=True
    )

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
                        os.path.join(
                            save_dir,
                            f"epoch_{epoch:03d}",
                            f"sample_{sample_id:04d}.png"
                        ),
                        result
                    )

                sample_id += 1
