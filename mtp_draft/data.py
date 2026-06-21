from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from mtp_draft.config import MTPConfig


def build_prompt(
    example: dict,
    tokenizer,
    max_prompt_len: int,
) -> tuple[list[int], list[int]]:
    """Format a ToolAlpaca example into (prompt_ids, answer_ids).

    Uses the pre-formatted ``prompt`` field as the prompt and
    ``golden_response`` (list of response turns) as the answer.

    Returns:
        prompt_ids:  token ids for the prompt, truncated to max_prompt_len
        answer_ids:  token ids for the answer (no special tokens)
    """
    prompt_text = example.get("prompt", "")
    answer = example.get("golden_response", "")

    if isinstance(answer, list):
        answer = " ".join(str(t) for t in answer)

    if not prompt_text or not answer:
        return [], []

    prompt_ids = tokenizer.encode(prompt_text)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    answer_ids = tokenizer.encode(str(answer), add_special_tokens=False)
    return prompt_ids, answer_ids


class FeatureDataset(Dataset):
    """Dataset over pre-cached teacher features.

    Each item corresponds to one (example, anchor_position) pair.
    The anchor_position j ∈ {0, …, cache_n_answer_positions-1} indexes into
    the cached feature tensor for each example.

    Shard format (produced by cache.py / Task 8):
        features_int8 : Tensor(cache_n_answer_positions, n_layers, d_teacher) int8
        scale         : scalar float32 Tensor
        prompt_ids    : Tensor(prompt_len,) long
        answer_ids    : Tensor(answer_len,) long
        handoff       : torch.tensor(int, dtype=torch.long)

    Returns per item:
        hiddens      : (n_layers, d_teacher) bfloat16 — dequantised
        context_ids  : (max_prompt_len,) long — left-padded with 0
        targets      : (max_draft,) long — tokens after anchor; -100 for padding
        valid_len    : scalar long tensor — number of valid (non-padded) targets
    """

    def __init__(self, shard_paths: list[Path], cfg: MTPConfig) -> None:
        self.cfg = cfg
        # items: list of (entry_dict, anchor_index j)
        self.items: list[tuple[dict, int]] = []
        for path in shard_paths:
            shard: list[dict] = torch.load(path, weights_only=False)
            for entry in shard:
                n_positions = entry["features_int8"].shape[0]
                for j in range(n_positions):
                    self.items.append((entry, j))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        entry, j = self.items[idx]
        cfg = self.cfg

        # --- dequantise features ---
        f_int8 = entry["features_int8"][j].float()   # (n_layers, d_teacher)
        hiddens = (f_int8 * entry["scale"]).bfloat16()

        # --- context: prompt tokens up to handoff + j, left-padded ---
        handoff: int = int(entry["handoff"].item())
        full_ids = torch.cat([entry["prompt_ids"], entry["answer_ids"]])
        raw_ctx = full_ids[: handoff + j]
        L = len(raw_ctx)
        if L >= cfg.max_prompt_len:
            context_ids = raw_ctx[-cfg.max_prompt_len :]
        else:
            pad = torch.zeros(cfg.max_prompt_len - L, dtype=torch.long)
            context_ids = torch.cat([pad, raw_ctx])

        # --- targets: tokens immediately after anchor ---
        start = handoff + j + 1
        raw_targets = full_ids[start : start + cfg.max_draft]
        valid_len = len(raw_targets)
        if valid_len < cfg.max_draft:
            pad = torch.full((cfg.max_draft - valid_len,), -100, dtype=torch.long)
            targets = torch.cat([raw_targets, pad])
        else:
            targets = raw_targets

        return hiddens, context_ids, targets, torch.tensor(valid_len, dtype=torch.long)


def get_dataloaders(cfg: MTPConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from pre-cached shards.

    Expects shards named ``train_shard_*.pt`` and ``validation_shard_*.pt``
    inside ``cfg.cache_dir``.
    """
    cache = Path(cfg.cache_dir)
    train_shards = sorted(cache.glob("train_shard_*.pt"))
    val_shards = sorted(cache.glob("validation_shard_*.pt"))

    train_ds = FeatureDataset(train_shards, cfg)
    val_ds = FeatureDataset(val_shards, cfg)

    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_loader, val_loader
