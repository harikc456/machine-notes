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
    """Format a HotpotQA example into (prompt_ids, answer_ids).

    Prompt format:
        Question: {q}\\n\\nContext:\\n{para_1}\\n...{para_n}\\n\\nAnswer:

    Context paragraphs are truncated (last paragraphs dropped first) to fit
    within max_prompt_len. The question is always preserved.

    Returns:
        prompt_ids:  token ids for the full prompt (len <= max_prompt_len)
        answer_ids:  token ids for the answer string (no special tokens)
    """
    question = example["question"]
    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]
    answer = example["answer"]

    q_prefix = f"Question: {question}\n\nContext:\n"
    a_suffix = "\n\nAnswer:"

    q_ids = tokenizer.encode(q_prefix)
    a_sep_ids = tokenizer.encode(a_suffix)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)

    budget = max_prompt_len - len(q_ids) - len(a_sep_ids) - 2

    ctx_ids: list[int] = []
    for title, sent_list in zip(titles, sentences):
        para = title + ": " + " ".join(sent_list) + "\n"
        para_ids = tokenizer.encode(para, add_special_tokens=False)
        if len(ctx_ids) + len(para_ids) > budget:
            break
        ctx_ids.extend(para_ids)

    prompt_ids = q_ids + ctx_ids + a_sep_ids
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
        raw_ctx = entry["prompt_ids"][: handoff + j]
        L = len(raw_ctx)
        if L >= cfg.max_prompt_len:
            context_ids = raw_ctx[-cfg.max_prompt_len :]
        else:
            pad = torch.zeros(cfg.max_prompt_len - L, dtype=torch.long)
            context_ids = torch.cat([pad, raw_ctx])

        # --- targets: tokens immediately after anchor ---
        full_ids = torch.cat([entry["prompt_ids"], entry["answer_ids"]])
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
