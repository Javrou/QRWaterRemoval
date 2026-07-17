from models.restormer import Restormer


def build_model(cfg, device):
    model = Restormer(
        inp_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        dim=cfg.dim,
        num_blocks=cfg.num_blocks,
        num_refinement_blocks=cfg.num_refinement_blocks,
        heads=cfg.heads,
        ffn_expansion_factor=cfg.ffn_expansion_factor,
        bias=False,
        LayerNorm_type="WithBias"
    ).to(device)

    return model
