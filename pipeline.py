from configs.pretrain import PretrainConfig
from configs.finetune import FinetuneConfig
from datasets.builder import build_pretrain_loader, build_mixed_loader
from engine.experiment import run_experiment

if __name__ == "__main__":
    # ============ Phase 1: Pretrain ============
    print("=" * 50)
    print("  Phase 1: Pretrain (Synthetic)")
    print("=" * 50)
    run_experiment(PretrainConfig, build_pretrain_loader, "pretrain")

    # ============ Phase 2: Finetune ============
    print("\n" + "=" * 50)
    print("  Phase 2: Finetune (Real)")
    print("=" * 50)
    run_experiment(FinetuneConfig, build_mixed_loader, "finetune", use_ema=True)

    print("\nPipeline finished.")
