# idlm/tests/test_config.py
import pytest
from pathlib import Path
from idlm.config import IDLMConfig, load_config


def test_defaults():
    cfg = IDLMConfig(ar_checkpoint="dummy.pt")
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16.0
    assert cfg.lora_target_modules == ["q_proj", "v_proj"]
    assert cfg.seq_len == 512
    assert cfg.batch_size == 8
    assert cfg.max_steps == 10_000
    assert cfg.lr == 3e-4
    assert cfg.warmup_steps == 200
    assert cfg.grad_clip == 1.0
    assert cfg.seed == 42
    assert cfg.eval_every == 500
    assert cfg.stride == 4
    assert cfg.num_eval_examples == 200
    assert cfg.prompt_len == 64
    assert cfg.gen_len == 128
    assert cfg.vocab_size == 50257


def test_load_config_from_yaml(tmp_path):
    yaml_text = "ar_checkpoint: /some/path.pt\nlora_rank: 4\n"
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml_text)
    cfg = load_config(p)
    assert cfg.ar_checkpoint == "/some/path.pt"
    assert cfg.lora_rank == 4
    assert cfg.seq_len == 512  # default preserved


def test_unknown_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("ar_checkpoint: x.pt\nunknown_key: 1\n")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(p)


def test_ar_checkpoint_required():
    with pytest.raises(TypeError):
        IDLMConfig()
