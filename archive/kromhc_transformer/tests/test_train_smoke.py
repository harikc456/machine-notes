import pytest
import torch
from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.train import train

def test_train_smoke():
    """Minimal smoke test: 1 epoch of training completes without errors."""
    cfg = KromHCConfig(
        d_model=64, n_heads=4, n_layers=1, ffn_hidden=128,
        batch_size=4, n_epochs=1, seq_len=64,
        model_type="kromhc", use_kromhc=True, qk_norm=True,
        seed=42,
        max_train_batches=20,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = train(cfg, device)
    assert "test_ppl" in metrics
    assert metrics["test_ppl"] > 1.0
    assert "test_loss" in metrics
    assert metrics["test_loss"] > 0.0
    assert isinstance(metrics["val_ppls"], list)
    assert len(metrics["val_ppls"]) == 1  # 1 epoch
