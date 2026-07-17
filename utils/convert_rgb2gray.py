import torch


def convert_rgb_to_gray(
        input_path,
        output_path
):
    print("=" * 50)
    print("Load:", input_path)

    checkpoint = torch.load(
        input_path,
        map_location="cpu",
        weights_only=False
    )

    state_dict = checkpoint["model"]

    new_state_dict = {}

    for k, v in state_dict.items():

        # =========================
        # RGB input -> Gray input
        # =========================
        if k == "patch_embed.proj.weight":

            print(
                "Convert:",
                k,
                v.shape,
                "->",
                end=" "
            )

            # [24,3,3,3]
            v = v.mean(
                dim=1,
                keepdim=True
            )

            print(v.shape)


        # =========================
        # RGB output -> Gray output
        # =========================
        elif k == "output.weight":

            print(
                "Convert:",
                k,
                v.shape,
                "->",
                end=" "
            )

            # [3,48,3,3]
            v = v.mean(
                dim=0,
                keepdim=True
            )

            print(v.shape)


        elif k == "output.bias":

            print(
                "Convert:",
                k,
                v.shape,
                "->",
                end=" "
            )

            # [3]
            v = v.mean(
                dim=0,
                keepdim=True
            )

            print(v.shape)

        new_state_dict[k] = v

    checkpoint["model"] = new_state_dict

    # =========================
    # EMA处理
    # =========================
    if "ema" in checkpoint:

        print("Convert EMA")

        ema_state = checkpoint["ema"]

        new_ema = {}

        for k, v in ema_state.items():

            if k == "patch_embed.proj.weight":

                v = v.mean(
                    dim=1,
                    keepdim=True
                )

            elif k == "output.weight":

                v = v.mean(
                    dim=0,
                    keepdim=True
                )

            elif k == "output.bias":

                v = v.mean(
                    dim=0,
                    keepdim=True
                )

            new_ema[k] = v

        checkpoint["ema"] = new_ema

    torch.save(
        checkpoint,
        output_path
    )

    print("=" * 50)
    print("Saved:")
    print(output_path)
    print("=" * 50)


if __name__ == "__main__":
    convert_rgb_to_gray(
        input_path=
        "../checkpoints/best_zxing.pth",

        output_path=
        "../checkpoints/best_zxing_gray.pth"
    )
