"""
Phase 1: extract teacher hidden states from ToolAlpaca and save int8 shards.

Usage:
    python -m mtp_draft.cache --config mtp_draft/configs/default.yaml --split train
    python -m mtp_draft.cache --config mtp_draft/configs/default.yaml --split validation
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
from tqdm import tqdm

from mtp_draft.config import MTPConfig, load_config
from mtp_draft.data import build_prompt


def _quantise_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor int8 quantisation. Returns (int8_tensor, scale)."""
    scale = t.float().abs().max() / 127.0
    if scale == 0:
        scale = torch.tensor(1.0)
    q = (t.float() / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def _dequantise_int8(t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantise_int8. Returns float32 tensor."""
    return t.float() * scale


def extract_and_cache(cfg: MTPConfig, split: str = "train") -> None:
    """
    Load Gemma 4 E2b, run over ToolAlpaca `split`, extract hidden states at
    cfg.teacher_layers for up to cfg.cache_n_answer_positions per example,
    quantise to int8, and write sharded .pt files to cfg.cache_dir.

    ToolAlpaca only has a train split; validation is carved out as the last 10%.

    Shard file format (list of dicts):
        {
            "features_int8": Tensor(cache_n_answer_positions, n_layers, d_teacher) int8,
            "scale":         Tensor scalar float32,
            "prompt_ids":    Tensor(prompt_len,) long,
            "answer_ids":    Tensor(answer_len,) long,
            "handoff":       torch.tensor(handoff, dtype=torch.long),
        }
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(cfg.cache_dir, exist_ok=True)

    raw = load_dataset("Ahren09/ToolAlpaca", split="train")
    splits = raw.train_test_split(test_size=0.1, seed=cfg.seed)
    dataset = splits["train"] if split == "train" else splits["test"]
    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id,
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
        device_map="cuda",
    ).eval()

    # teacher_layers are 0-indexed layer numbers;
    # hidden_states[0] = embedding output, hidden_states[k+1] = layer k output
    layer_indices = [l + 1 for l in cfg.teacher_layers]

    shard_data: list[dict] = []
    shard_idx = 0
    cache_dir = Path(cfg.cache_dir)

    for example in tqdm(dataset, desc=f"Extracting {split}"):
        prompt_ids, answer_ids = build_prompt(example, tokenizer, cfg.max_prompt_len)
        if not prompt_ids or not answer_ids:
            continue

        handoff = len(prompt_ids) - 1  # index of last prompt token (0-based)
        n_pos = min(cfg.cache_n_answer_positions, len(answer_ids))

        # Positions to cache: handoff, handoff+1, ..., handoff+n_pos-1
        # (handoff = last prompt token; handoff+k = k-th answer token)
        full_ids = prompt_ids + list(answer_ids[:n_pos])
        input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).cuda()

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1,) each (1, seq_len, d_model)
        all_hidden = outputs.hidden_states

        pos_features = []
        for k in range(n_pos):
            pos = handoff + k
            layer_feats = torch.stack([
                all_hidden[li][0, pos, :].float()
                for li in layer_indices
            ])  # (n_layers, d_teacher)
            pos_features.append(layer_feats)

        features = torch.stack(pos_features)  # (n_pos, n_layers, d_teacher)
        # Pad to cache_n_answer_positions if shorter
        if n_pos < cfg.cache_n_answer_positions:
            pad = torch.zeros(
                cfg.cache_n_answer_positions - n_pos,
                len(cfg.teacher_layers),
                cfg.d_teacher,
            )
            features = torch.cat([features, pad], dim=0)

        q_feat, scale = _quantise_int8(features)

        shard_data.append({
            "features_int8": q_feat.cpu(),
            "scale": scale.cpu(),
            "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "answer_ids": torch.tensor(list(answer_ids), dtype=torch.long),
            "handoff": torch.tensor(handoff, dtype=torch.long),
        })

        if len(shard_data) == cfg.cache_shard_size:
            path = cache_dir / f"{split}_shard_{shard_idx:04d}.pt"
            torch.save(shard_data, path)
            print(f"  Saved {path}")
            shard_data = []
            shard_idx += 1

    if shard_data:
        path = cache_dir / f"{split}_shard_{shard_idx:04d}.pt"
        torch.save(shard_data, path)
        print(f"  Saved {path}")

    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mtp_draft/configs/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    extract_and_cache(cfg, split=args.split)
