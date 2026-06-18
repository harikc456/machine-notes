from __future__ import annotations
import tempfile
from pathlib import Path
import torch
import pytest
from medusa.config import MedusaConfig
from medusa.data import MedusaDataset, get_dataloaders

D_MODEL = 32
N_HEADS = 3
N_POS = 5


@pytest.fixture
def cfg(tmp_path):
    return MedusaConfig(
        n_heads=N_HEADS,
        d_model=D_MODEL,
        batch_size=2,
        val_split=0.5,
        cache_dir=str(tmp_path),
        seed=42,
    )


def _make_shard(path: Path, n_entries: int = 4) -> None:
    shard = []
    for _ in range(n_entries):
        h = torch.randint(-127, 128, (N_POS, D_MODEL), dtype=torch.int8)
        scale = torch.rand(N_POS) + 0.01
        targets = torch.randint(0, 100, (N_POS, N_HEADS), dtype=torch.long)
        shard.append({"hidden_int8": h, "scale": scale, "targets": targets})
    torch.save(shard, path)


def test_dataset_len(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt", n_entries=3)
    ds = MedusaDataset([tmp_path / "train_shard_0000.pt"], cfg)
    assert len(ds) == 3 * N_POS


def test_dataset_item_shapes(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt")
    ds = MedusaDataset([tmp_path / "train_shard_0000.pt"], cfg)
    h, targets = ds[0]
    assert h.shape == (D_MODEL,)
    assert targets.shape == (N_HEADS,)


def test_dataset_dequantize(cfg, tmp_path):
    shard = [{
        "hidden_int8": torch.full((1, D_MODEL), 63, dtype=torch.int8),
        "scale": torch.tensor([2.0]),
        "targets": torch.zeros(1, N_HEADS, dtype=torch.long),
    }]
    path = tmp_path / "train_shard_0000.pt"
    torch.save(shard, path)
    ds = MedusaDataset([path], cfg)
    h, _ = ds[0]
    assert h.dtype == torch.float32
    assert torch.allclose(h, torch.full((D_MODEL,), 126.0))


def test_dataset_targets_dtype(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt")
    ds = MedusaDataset([tmp_path / "train_shard_0000.pt"], cfg)
    _, targets = ds[0]
    assert targets.dtype == torch.long


def test_get_dataloaders(cfg, tmp_path):
    _make_shard(tmp_path / "train_shard_0000.pt", n_entries=4)
    _make_shard(tmp_path / "validation_shard_0000.pt", n_entries=2)
    train_dl, val_dl = get_dataloaders(cfg)
    h, t = next(iter(train_dl))
    assert h.shape == (2, D_MODEL)
    assert t.shape == (2, N_HEADS)
