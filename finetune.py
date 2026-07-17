from configs.finetune import FinetuneConfig
from datasets.builder import build_mixed_loader
from engine.experiment import run_experiment

if __name__ == "__main__":
    run_experiment(FinetuneConfig, build_mixed_loader, "finetune", use_ema=True)
