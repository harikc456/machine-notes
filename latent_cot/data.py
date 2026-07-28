from __future__ import annotations
import re
import random

_CALC_RE = re.compile(r"<<[^>]*>>")
_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")


def split_answer(answer: str) -> tuple[str, str]:
    """Split a GSM8K `answer` field into (reasoning_trace, normalized_label)."""
    if "####" not in answer:
        raise ValueError(f"answer has no '####' delimiter: {answer[:80]!r}")
    trace, _, tail = answer.partition("####")
    return trace.strip(), normalize_number(tail)


def strip_calc_annotations(trace: str) -> str:
    """Remove `<<expr=val>>` calculator spans, leaving natural-language reasoning."""
    return _CALC_RE.sub("", trace)


def normalize_number(s: str) -> str:
    """Canonicalize a numeric string: drop $, commas, whitespace, trailing .0."""
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    m = _NUM_RE.search(s)
    if not m:
        return s.strip()
    val = m.group(0).replace(",", "")
    if "." in val:
        val = val.rstrip("0").rstrip(".")
    return val


def shuffle_trace(trace: str, rng: random.Random) -> str:
    """Reorder the reasoning steps. Splits on newlines; falls back to sentences."""
    lines = [ln for ln in trace.splitlines() if ln.strip()]
    if len(lines) < 2:
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", trace) if s.strip()]
    rng.shuffle(lines)
    return "\n".join(lines)


def load_gsm8k(split: str, strip_annotations: bool = True) -> list[dict]:
    """Load GSM8K `main` split as a list of {question, trace, label} dicts."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rows: list[dict] = []
    for ex in ds:
        trace, label = split_answer(ex["answer"])
        if strip_annotations:
            trace = strip_calc_annotations(trace)
        rows.append({"question": ex["question"].strip(), "trace": trace, "label": label})
    return rows


import random as _random
import torch
from torch.utils.data import Dataset


class GSM8KDataset(Dataset):
    """Thin wrapper over a list of {question, trace, label} dicts."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _pad(seqs: list[list[int]], pad_val: int, pad_side: str = "right") -> torch.Tensor:
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_val, dtype=torch.long)
    for i, s in enumerate(seqs):
        if pad_side == "left":
            out[i, maxlen - len(s):] = torch.tensor(s, dtype=torch.long)
        else:
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def _mask_from(seqs: list[list[int]], maxlen: int, pad_side: str = "right") -> torch.Tensor:
    m = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        if pad_side == "left":
            m[i, maxlen - len(s):] = 1
        else:
            m[i, : len(s)] = 1
    return m


class Collator:
    """Tokenizes a batch of rows per condition.

    include_answer=True  -> training batch (answer tokens supervised).
        Padded right: teacher-forcing over the full sequence with -100
        masked labels doesn't require any particular padding side.
    include_answer=False -> eval batch (prefix only; generate the answer).
        Padded left: batched causal-LM generation (see `generate()` in
        model.py) needs every row's last real token at the same rightmost
        position so the next-token-to-generate lines up across the batch.
    """

    def __init__(self, tokenizer, cfg, condition: str, include_answer: bool):
        self.tok = tokenizer
        self.cfg = cfg
        self.condition = condition
        self.include_answer = include_answer
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id

    def _enc(self, text: str, max_len: int, add_special: bool) -> list[int]:
        return self.tok(
            text, add_special_tokens=add_special, truncation=True, max_length=max_len
        )["input_ids"]

    def _answer_ids(self, label: str) -> list[int]:
        ids = self.tok(" " + label, add_special_tokens=False)["input_ids"]
        return (ids + [self.eos_id])[: self.cfg.max_answer_tokens]

    def _recon_ids(self, trace: str) -> list[int]:
        ids = self.tok(trace, add_special_tokens=False)["input_ids"]
        return (ids + [self.eos_id])[: self.cfg.max_trace_tokens]

    def __call__(self, rows: list[dict]) -> dict:
        c = self.condition
        batch: dict = {"label_text": [r["label"] for r in rows]}

        if c in ("floor", "ceiling"):
            prompts, fulls, labels = [], [], []
            for r in rows:
                if c == "floor":
                    ptext = f"{r['question']}\nAnswer:"
                else:
                    ptext = f"{r['question']}\nReasoning: {r['trace']}\nAnswer:"
                p_ids = self._enc(ptext, self.cfg.max_trace_tokens
                                  + self.cfg.max_question_tokens, add_special=True)
                if self.include_answer:
                    a_ids = self._answer_ids(r["label"])
                    fulls.append(p_ids + a_ids)
                    labels.append([-100] * len(p_ids) + a_ids)
                else:
                    prompts.append(p_ids)
            if self.include_answer:
                input_ids = _pad(fulls, self.pad_id)
                batch["input_ids"] = input_ids
                batch["attention_mask"] = _mask_from(fulls, input_ids.size(1))
                batch["labels"] = _pad(labels, -100)
            else:
                input_ids = _pad(prompts, self.pad_id, pad_side="left")
                batch["input_ids"] = input_ids
                batch["attention_mask"] = _mask_from(
                    prompts, input_ids.size(1), pad_side="left"
                )
            return batch

        if c == "reconstruct":
            q_ids, recon_ids = [], []
            for r in rows:
                q_ids.append(self._enc(f"{r['question']}\nAnswer:",
                                       self.cfg.max_question_tokens, add_special=True))
                recon_ids.append(self._recon_ids(r["trace"]))
            qi = _pad(q_ids, self.pad_id)
            ri = _pad(recon_ids, self.pad_id)
            batch["question_ids"] = qi
            batch["question_mask"] = _mask_from(q_ids, qi.size(1))
            batch["recon_ids"] = ri
            batch["recon_mask"] = _mask_from(recon_ids, ri.size(1))
            return batch

        # z / z_shuffled
        rng = _random.Random(self.cfg.seed)
        trace_ids, q_ids, a_ids_list = [], [], []
        for r in rows:
            trace = r["trace"]
            if c == "z_shuffled":
                trace = shuffle_trace(trace, rng)
            trace_ids.append(self._enc(trace, self.cfg.max_trace_tokens, add_special=True))
            q_ids.append(self._enc(f"{r['question']}\nAnswer:",
                                   self.cfg.max_question_tokens, add_special=True))
            if self.include_answer:
                a_ids_list.append(self._answer_ids(r["label"]))

        # Trace padding side doesn't matter: `_encode_z` cross-attends into it
        # with a key_padding_mask, so pad tokens are masked out regardless of
        # where they sit. The question segment, however, feeds straight into
        # generation (z_prefix + question -> next token), so in eval mode it
        # must be left-padded for the same reason as the floor/ceiling path.
        q_pad_side = "right" if self.include_answer else "left"
        ti = _pad(trace_ids, self.pad_id)
        qi = _pad(q_ids, self.pad_id, pad_side=q_pad_side)
        batch["trace_ids"] = ti
        batch["trace_mask"] = _mask_from(trace_ids, ti.size(1))
        batch["question_ids"] = qi
        batch["question_mask"] = _mask_from(q_ids, qi.size(1), pad_side=q_pad_side)
        if self.include_answer:
            ai = _pad(a_ids_list, self.pad_id)
            batch["answer_ids"] = ai
            batch["answer_mask"] = _mask_from(a_ids_list, ai.size(1))
        return batch
