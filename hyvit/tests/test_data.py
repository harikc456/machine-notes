import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from config import HyViTTinyConfig
from data.cifar import build_loaders


def test_loader_shapes():
    cfg = HyViTTinyConfig()
    cfg.batch_size = 4
    train_loader, val_loader = build_loaders(cfg, num_workers=0)
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 32, 32)
    assert labels.shape == (4,)
    assert images.dtype == torch.float32


def test_loader_label_range():
    cfg = HyViTTinyConfig()
    cfg.batch_size = 128
    _, val_loader = build_loaders(cfg, num_workers=0)
    for images, labels in val_loader:
        assert labels.min() >= 0 and labels.max() <= 9
        break


def test_train_val_different():
    """Train and val should return different data (different subsets + augmentations)."""
    cfg = HyViTTinyConfig()
    cfg.batch_size = 4
    train_loader, val_loader = build_loaders(cfg, num_workers=0)
    t_imgs, _ = next(iter(train_loader))
    v_imgs, _ = next(iter(val_loader))
    # Val loader uses batch_size*2; compare only the first 4 images
    assert not torch.allclose(t_imgs, v_imgs[:4])
