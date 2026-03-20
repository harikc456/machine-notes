from __future__ import annotations
import numpy as np
import pytest
import torch
from PIL import Image
from unittest.mock import patch

from flow_matching.config import FlowConfig


class _FakeCIFAR100:
    """Minimal CIFAR-100 stub — avoids network download in tests."""
    def __init__(self, *args, **kwargs):
        self.data = np.random.randint(0, 256, (50, 32, 32, 3), dtype=np.uint8)
        self.targets = list(range(50))
        self.transform = kwargs.get("transform")

    def __len__(self) -> int:
        return 50

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx] % 100


def test_build_loaders_batch_shape(tmp_path):
    from flow_matching.data import build_loaders

    cfg = FlowConfig(data_root=str(tmp_path), batch_size=4)
    with patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100):
        train_loader, val_loader = build_loaders(cfg, num_workers=0)

    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32), f"Expected (4,3,32,32), got {x.shape}"
    assert x.dtype == torch.float32
    assert 0 <= int(y.min()) and int(y.max()) <= 99


def test_build_loaders_returns_two_loaders(tmp_path):
    from flow_matching.data import build_loaders
    from torch.utils.data import DataLoader

    cfg = FlowConfig(data_root=str(tmp_path), batch_size=4)
    with patch("flow_matching.data.torchvision.datasets.CIFAR100", _FakeCIFAR100):
        result = build_loaders(cfg, num_workers=0)

    assert len(result) == 2
    train_loader, val_loader = result
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
