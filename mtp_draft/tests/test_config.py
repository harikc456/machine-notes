import pytest
from mtp_draft.config import MTPConfig, load_config
import tempfile, yaml, os

def test_default_config_valid():
    cfg = MTPConfig()
    assert cfg.d_draft == 512
    assert cfg.n_blocks == 4
    assert cfg.ffn_hidden == 1366
    assert len(cfg.teacher_layers) == 4

def test_ffn_hidden_auto():
    cfg = MTPConfig(d_draft=384, ffn_hidden=0)
    assert cfg.ffn_hidden == int(8 / 3 * 384)

def test_teacher_layers_power_of_two():
    with pytest.raises(AssertionError, match="power of 2"):
        MTPConfig(teacher_layers=[3, 8, 17])  # 3 layers = not power of 2

def test_load_config_roundtrip():
    cfg = MTPConfig(d_draft=256, n_blocks=2)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"d_draft": 256, "n_blocks": 2}, f)
        path = f.name
    loaded = load_config(path)
    os.unlink(path)
    assert loaded.d_draft == 256
    assert loaded.n_blocks == 2
