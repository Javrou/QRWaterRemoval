import torch
import zxingcpp

from model.restormer import Restormer
from tool.pretrain_data_loader import test_loader
from tool.checkpoints import load_checkpoint


def main():
    # ======================
    # Device
    # ======================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ======================
    # Model
    # ======================
    model = Restormer(
        inp_channels=1,
        out_channels=1,
        dim=24,
        num_blocks=[2, 2, 2, 3],
        num_refinement_blocks=1,
        heads=[1, 2, 2, 4],
        ffn_expansion_factor=2.0,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(device)

    # ======================
    # Load checkpoint
    # ======================
    ckpt_path = "checkpoints/best.pth"

    epoch, step, best_metrics = load_checkpoint(
        ckpt_path,
        model,
        optimizer=None,
        scheduler=None,
        device=device
    )
    print(f"Epoch : {epoch}")
    print(f"Step  : {step}")
    for k, v in best_metrics.items():
        print(f"{k:12}: {v:.6f}")

    model.eval()

    # ======================
    # ZXing
    # ======================
    def zxing_rate(batch):
        batch = batch.detach().cpu()

        success = 0
        total = batch.shape[0]

        for i in range(total):
            img = batch[i].permute(1, 2, 0).numpy()
            img = (img * 255).clip(0, 255).astype("uint8")

            if len(zxingcpp.read_barcodes(img)) > 0:
                success += 1

        return success / total

    # ======================
    # TEST LOOP
    # ======================
    total = 0
    success = 0

    with torch.no_grad():
        for i, (inp, tgt) in enumerate(test_loader):

            inp = inp.to(device)
            tgt = tgt.to(device)

            pred = model(inp).clamp(0, 1)

            sr = zxing_rate(pred)

            success += sr
            total += 1

            if (i + 1) % 10 == 0:
                print(f"[{i + 1}/{len(test_loader)}] ZXing={sr:.4f}")

    # ======================
    # RESULT
    # ======================
    print("\n====================")
    print(f"Final ZXing Rate: {success / total:.4f}")
    print("====================")


if __name__ == '__main__':
    main()
