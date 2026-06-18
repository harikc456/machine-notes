"""
Phase 1: extract Gemma 4 E2B last hidden states from ShareGPT52K and save int8 shards.

Usage:
    python -m medusa.cache --config medusa/configs/default.yaml --split train
    python -m medusa.cache --config medusa/configs/default.yaml --split validation
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from medusa.config import MedusaConfig, load_config


def _quantise_int8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-vector int8 quantization for (n_pos, d_model) tensor.

    Returns:
        q_t:   (n_pos, d_model) int8
        scales: (n_pos,) float32 — one scale per row
    """
    scales = t.float().abs().amax(dim=-1).clamp(min=1e-8) / 127.0
    q_t = (t.float() / scales.unsqueeze(-1)).clamp(-127, 127).to(torch.int8)
    return q_t, scales.float()


def _dequantise_int8(q_t: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Inverse of _quantise_int8. Returns float32 tensor."""
    return q_t.float() * scales.unsqueeze(-1)


def _extract_conversation(
    example: dict,
    tokenizer,
    cfg: MedusaConfig,
) -> tuple[list[int], list[int]]:
    """Extract the first human→gpt exchange from a ShareGPT52K conversation.

    Returns (prompt_ids, answer_ids). Both empty on failure.
    """
    convs = example.get("conversations", [])
    human_turn = next((c for c in convs if c.get("from") == "human"), None)
    gpt_turn = next((c for c in convs if c.get("from") == "gpt"), None)
    if not human_turn or not gpt_turn:
        return [], []

    human_text = human_turn.get("value", "").strip()
    gpt_text = gpt_turn.get("value", "").strip()
    if not human_text or not gpt_text:
        return [], []

    messages = [{"role": "user", "content": human_text}]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt_text = human_text

    prompt_ids = tokenizer.encode(prompt_text)
    answer_ids = tokenizer.encode(gpt_text, add_special_tokens=False)

    # Truncate so total sequence fits within max_seq_len
    max_prompt = cfg.max_seq_len - min(cfg.max_answer_len, len(answer_ids))
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    return prompt_ids, answer_ids


def extract_and_cache(cfg: MedusaConfig, split: str = "train") -> None:
    """Run teacher on ShareGPT52K, extract last hidden states, write int8 shards."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM

    os.makedirs(cfg.cache_dir, exist_ok=True)

    raw = load_dataset(cfg.dataset_id, split="train")
    splits = raw.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    dataset = splits["train"] if split == "train" else splits["test"]  # HF names this "test"

    tokenizer = AutoTokenizer.from_pretrained(cfg.teacher_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.teacher_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    shard_data: list[dict] = []
    shard_idx = 0
    cache_dir = Path(cfg.cache_dir)

    for example in tqdm(dataset, desc=f"Caching {split}"):
        prompt_ids, answer_ids = _extract_conversation(example, tokenizer, cfg)
        if not prompt_ids or not answer_ids:
            continue

        n_pos = min(cfg.max_answer_len, len(answer_ids))
        # Full sequence: prompt + first n_pos answer tokens
        full_ids = prompt_ids + list(answer_ids[:n_pos])
        input_ids = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).cuda()

        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            last_hidden = out.hidden_states[-1][0].float().cpu()  # (seq_len, d_model)

        handoff = len(prompt_ids) - 1  # index of last prompt token

        # Extract positions: handoff, handoff+1, ..., handoff+n_pos-1
        # These are the positions just before each answer token is emitted
        positions = list(range(handoff, handoff + n_pos))
        hidden_vecs = last_hidden[positions]  # (n_pos, d_model)

        # Build targets: for position p, targets[k] = full_ids[p + k + 1]
        full_t = torch.tensor(full_ids, dtype=torch.long)
        targets = torch.full((n_pos, cfg.n_heads), -100, dtype=torch.long)
        for i, pos in enumerate(positions):
            for k in range(cfg.n_heads):
                tgt_idx = pos + k + 1
                if tgt_idx < len(full_t):
                    targets[i, k] = full_t[tgt_idx]

        q_hidden, scales = _quantise_int8(hidden_vecs)

        shard_data.append({
            "hidden_int8": q_hidden.cpu(),
            "scale": scales.cpu(),
            "targets": targets.cpu(),
        })

        if len(shard_data) >= cfg.cache_shard_size:
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
    parser.add_argument("--config", default="medusa/configs/default.yaml")
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    args = parser.parse_args()
    extract_and_cache(load_config(args.config), split=args.split)
