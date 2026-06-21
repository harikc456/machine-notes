from __future__ import annotations
import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader

from medusa.config import MedusaConfig


class MedusaDataset(IterableDataset):
    """Lazy-loading dataset over pre-cached Medusa hidden-state shards.

    Shard format (list of dicts):
        hidden_int8 : Tensor(n_pos, d_model) int8
        scale       : Tensor(n_pos,) float32  — per-vector scale
        targets     : Tensor(n_pos, n_heads) int64  — -100 for padding

    Yields pre-formed batches of shape (B, d_model) / (B, n_heads) so the
    DataLoader can be used with batch_size=None — eliminating the Python
    collation overhead of 10M individual sample yields per epoch.
    """

    def __init__(self, shard_paths: list[Path], cfg: MedusaConfig, shuffle: bool = False) -> None:
        self.shard_paths = list(shard_paths)
        self.cfg = cfg
        self.shuffle = shuffle
        self._total = 0
        for path in self.shard_paths:
            shard: list[dict] = torch.load(path, weights_only=False)
            for entry in shard:
                assert entry["hidden_int8"].shape[-1] == cfg.d_model, (
                    f"Shard {path}: hidden dim {entry['hidden_int8'].shape[-1]} != cfg.d_model {cfg.d_model}"
                )
                self._total += entry["hidden_int8"].shape[0]
            del shard

    def __len__(self) -> int:
        return self._total // self.cfg.batch_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        paths = list(self.shard_paths)
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]
        if self.shuffle:
            random.shuffle(paths)

        B = self.cfg.batch_size
        buf_h: list[torch.Tensor] = []
        buf_t: list[torch.Tensor] = []
        buf_size = 0

        for path in paths:
            shard: list[dict] = torch.load(path, weights_only=False)
            entry_order = list(range(len(shard)))
            if self.shuffle:
                random.shuffle(entry_order)
            for idx in entry_order:
                entry = shard[idx]
                # Bulk dequantize entry at once
                h = entry["hidden_int8"].float() * entry["scale"].unsqueeze(-1)
                buf_h.append(h)
                buf_t.append(entry["targets"])
                buf_size += h.shape[0]

                # Yield complete batches once we've buffered enough positions
                if buf_size >= B:
                    cat_h = torch.cat(buf_h)   # (buf_size, d_model)
                    cat_t = torch.cat(buf_t)   # (buf_size, n_heads)
                    if self.shuffle:
                        perm = torch.randperm(cat_h.shape[0])
                        cat_h = cat_h[perm]
                        cat_t = cat_t[perm]
                    for start in range(0, cat_h.shape[0] - B + 1, B):
                        yield cat_h[start : start + B], cat_t[start : start + B]
                    remainder = cat_h.shape[0] % B
                    if remainder:
                        buf_h = [cat_h[-remainder:]]
                        buf_t = [cat_t[-remainder:]]
                        buf_size = remainder
                    else:
                        buf_h = []
                        buf_t = []
                        buf_size = 0
            del shard

        # Drop leftover positions — partial batches trigger recompilation with torch.compile


def get_dataloaders(cfg: MedusaConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and val DataLoaders from pre-cached shards."""
    cache = Path(cfg.cache_dir)
    train_shards = sorted(cache.glob("train_shard_*.pt"))
    val_shards = sorted(cache.glob("validation_shard_*.pt"))

    train_ds = MedusaDataset(train_shards, cfg, shuffle=True)
    val_ds = MedusaDataset(val_shards, cfg, shuffle=False)

    # batch_size=None: dataset yields pre-formed batches, no DataLoader collation
    train_loader = DataLoader(
        train_ds, batch_size=None, num_workers=cfg.num_workers,
        pin_memory=True, prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=None, num_workers=cfg.num_workers,
        pin_memory=True, prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader
