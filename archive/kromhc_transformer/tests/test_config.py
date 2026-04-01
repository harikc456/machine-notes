import pytest
from pathlib import Path
from kromhc_transformer.config import KromHCConfig, load_config

def test_config_defaults():
    cfg = KromHCConfig()
    assert cfg.d_model == 256
    assert cfg.n_heads == 8
    assert cfg.n_layers == 6
    assert cfg.model_type == "kromhc"
    assert cfg.use_kromhc == True
    assert cfg.qk_norm == True
    assert cfg.vocab_size == 50257
    assert cfg.seq_len == 512

def test_load_config_from_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("d_model: 512\nn_heads: 16\nmodel_type: baseline\n")
    cfg = load_config(yaml_file)
    assert cfg.d_model == 512
    assert cfg.n_heads == 16
    assert cfg.model_type == "baseline"
    assert cfg.use_kromhc == True  # default unchanged
