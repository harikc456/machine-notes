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
    ds = load_dataset("gsm8k", "main", split=split)
    rows: list[dict] = []
    for ex in ds:
        trace, label = split_answer(ex["answer"])
        if strip_annotations:
            trace = strip_calc_annotations(trace)
        rows.append({"question": ex["question"].strip(), "trace": trace, "label": label})
    return rows
