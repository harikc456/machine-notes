"""
Training entry point for RBF-FFN ablation experiments.

Usage:
    python -m rbf_ffn.train --config rbf_ffn/configs/g0_baseline.yaml
    python -m rbf_ffn.train --config rbf_ffn/configs/g2_sinkhorn.yaml --n_epochs 10

Each run saves results to:
    rbf_ffn/experiments/<date>_<gate_variant>_d<d_model>_K<K>/
        config.yaml      # copy of the config used for this run
        metrics.jsonl    # one JSON line per epoch: {"epoch": 0, "loss": 1.23, ...}
"""
import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rbf_ffn.config import RBFFFNConfig, load_config
from rbf_ffn.models.transformer_block import RBFTransformerBlock


def get_experiment_dir(cfg: RBFFFNConfig) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{cfg.gate_variant}_d{cfg.d_model}_K{cfg.K}"
    path = Path(__file__).parent / "experiments" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_toy_dataset(cfg: RBFFFNConfig, n_samples: int = 1024):
    """Toy random token classification dataset for smoke-testing."""
    x = torch.randn(n_samples, cfg.seq_len, cfg.d_model)
    y = torch.randint(0, cfg.num_classes, (n_samples,))
    return TensorDataset(x, y)


def train(cfg: RBFFFNConfig, config_path: Path, n_epochs: int = 5, batch_size: int = 32):
    exp_dir = get_experiment_dir(cfg)

    # Copy the source config file into the experiment directory for full reproducibility
    shutil.copy(config_path, exp_dir / "config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blocks = nn.ModuleList([RBFTransformerBlock(cfg) for _ in range(cfg.n_layers)])
    head = nn.Linear(cfg.d_model, cfg.num_classes)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = blocks
            self.head = head
            self.norm = nn.LayerNorm(cfg.d_model)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.head(self.norm(x[:, 0]))  # cls-token pool

    model = Model().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    dataset = build_toy_dataset(cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    metrics_path = exp_dir / "metrics.jsonl"
    print(f"Experiment dir: {exp_dir}")
    print(f"Config: {config_path}")

    for epoch in range(n_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(-1) == y_batch).sum().item()
            total += len(y_batch)

        avg_loss = total_loss / total
        acc = correct / total
        row = {"epoch": epoch, "loss": avg_loss, "acc": acc}
        print(row)
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"Done. Metrics saved to {metrics_path}")
    return exp_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train RBF-FFN transformer")
    parser.add_argument(
        "--config", required=True,
        help="Path to a YAML config file (e.g. rbf_ffn/configs/g0_baseline.yaml)"
    )
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path)
    train(cfg, config_path=config_path, n_epochs=args.n_epochs, batch_size=args.batch_size)
