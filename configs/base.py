class BaseExperimentConfig:
    # ========= Dataset =========
    gray = True
    num_workers = 8

    # ========= Model =========
    in_channels = 1
    out_channels = 1
    dim = 24
    num_blocks = [2, 2, 2, 3]
    num_refinement_blocks = 1
    heads = [1, 2, 2, 4]
    ffn_expansion_factor = 2.0

    # ========= Optimizer =========
    min_lr = 1e-6
    warmup_epochs = 5
    weight_decay = 1e-4
    lr_patience = 3
    grad_clip = 4

    # ========= Log =========
    print_freq = 10
    zxing_freq = 100

    # ========= Early Stop =========
    min_epochs = 15
    min_delta_loss = 1e-3

    # ========= Visual =========
    save_visual = True
    visual_num = 8

    # ========= Random =========
    seed = 42
