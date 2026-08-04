class BaseExperimentConfig:
    # ========= Dataset =========
    gray = True
    num_workers = 16

    # ========= Model =========
    in_channels = 1
    out_channels = 1
    dim = 36
    num_blocks = [3, 3, 4, 4]
    num_refinement_blocks = 1
    heads = [1, 2, 4, 4]
    ffn_expansion_factor = 2.0

    # ========= Optimizer =========
    min_lr = 1e-6
    warmup_epochs = 10
    weight_decay = 1e-4
    lr_patience = 3
    grad_clip = 4
    T0 = 10
    T_mult = 2

    # ========= Log =========
    print_freq = 20
    zxing_freq = 200

    # ========= Early Stop =========
    min_epochs = 15
    min_delta_loss = 1e-3

    # ========= Visual =========
    save_visual = True
    vis_freq = 2
    val_samples = 380
    visual_num = 8

    # ========= Random =========
    seed = 42
