from __future__ import annotations
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from medusa.config import MedusaConfig


class MedusaDataset(Dataset):
    """Dataset over pre-cached Medusa hidden-state shards.

    Shard format (list of dicts):
        hidden_int8 : Tensor(n_pos, d_model) int8
        scale       : Tensor(n_pos,) float32  — per-vector scale
        targets     : Tensor(n_pos, n_heads) int64  — -100 for padding

    Returns per item:
        hidden  : Tensor(d_model,) float32 — dequantized
        targets : Tensor(n_heads,) int64
    """

    def __init__(self, shard_paths: list[Path], cfg: MedusaConfig) -> None:
        self.cfg = cfg
        self.items: list[tuple[dict, int]] = []
        for path in shard_paths:
            shard: list[dict] = torch.load(path, weights_only=False)
            for entry in shard:
                assert entry["hidden_int8"].shape[-1] == cfg.d_model, (
                    f"Shard {path}: hidden dim {entry['hidden_int8'].shape[-1]} != cfg.d_model {cfg.d_model}"
                )
                n_pos = entry["hidden_int8"].shape[0]
                for i in range(n_pos):
                    self.items.append((entry, i))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        entry, i = self.items[idx]
        hidden = entry["hidden_int8"][i].float() * entry["scale"][i]
        targets = entry["targets"][i]
        return hidden, targets


def get_dataloaders(cfg: MedusaConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from pre-cached shards.

    Expects shards named ``train_shard_*.pt`` and ``validation_shard_*.pt``
    inside ``cfg.cache_dir``.
    """
    cache = Path(cfg.cache_dir)
    train_shards = sorted(cache.glob("train_shard_*.pt"))
    val_shards = sorted(cache.glob("validation_shard_*.pt"))

    train_ds = MedusaDataset(train_shards, cfg)
    val_ds = MedusaDataset(val_shards, cfg)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader
