from configs.pretrain import PretrainConfig
from datasets.builder import build_pretrain_loader
from engine.experiment import run_experiment

if __name__ == "__main__":
    run_experiment(PretrainConfig, build_pretrain_loader, "pretrain")
