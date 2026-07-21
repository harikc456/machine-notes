import random
from latent_cot.data import (
    split_answer, strip_calc_annotations, normalize_number, shuffle_trace,
)

RAW = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
    "#### 72"
)


def test_split_answer():
    trace, label = split_answer(RAW)
    assert label == "72"
    assert "####" not in trace
    assert "24 clips in May" in trace


def test_strip_calc_annotations():
    trace, _ = split_answer(RAW)
    clean = strip_calc_annotations(trace)
    assert "<<" not in clean and ">>" not in clean
    assert "24 clips in May" in clean  # surrounding text preserved


def test_normalize_number():
    assert normalize_number(" $1,000 ") == "1000"
    assert normalize_number("72") == "72"
    assert normalize_number("5.0") == "5"
    assert normalize_number("-3") == "-3"


def test_shuffle_trace_reorders_but_preserves_steps():
    trace, _ = split_answer(RAW)
    clean = strip_calc_annotations(trace)
    rng = random.Random(0)
    shuffled = shuffle_trace(clean, rng)
    # same set of non-empty lines, (very likely) different order
    assert sorted(shuffled.splitlines()) == sorted(clean.splitlines())


import pytest
import torch
from latent_cot.config import ExperimentConfig
from latent_cot.data import GSM8KDataset, Collator, _pad, _mask_from


def test_pad_right_default_places_content_at_start():
    seqs = [[1, 2, 3], [4, 5]]
    out = _pad(seqs, pad_val=0)
    assert out.tolist() == [[1, 2, 3], [4, 5, 0]]
    mask = _mask_from(seqs, maxlen=3)
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_pad_left_places_content_at_end():
    seqs = [[1, 2, 3], [4, 5]]
    out = _pad(seqs, pad_val=0, pad_side="left")
    assert out.tolist() == [[1, 2, 3], [0, 4, 5]]
    mask = _mask_from(seqs, maxlen=3, pad_side="left")
    assert mask.tolist() == [[1, 1, 1], [0, 1, 1]]

_ROWS = [
    {"question": "Q1?", "trace": "step one\nstep two", "label": "72"},
    {"question": "A longer question here?", "trace": "only one line", "label": "5"},
]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")


@pytest.mark.slow
@pytest.mark.parametrize("condition", ["floor", "ceiling", "z", "z_shuffled"])
def test_collator_shapes(tok, condition):
    cfg = ExperimentConfig(condition=condition)
    coll = Collator(tok, cfg, condition, include_answer=True)
    batch = coll(_ROWS)
    B = len(_ROWS)
    assert batch["label_text"] == ["72", "5"]
    if condition in ("floor", "ceiling"):
        for k in ("input_ids", "attention_mask", "labels"):
            assert batch[k].shape[0] == B and batch[k].ndim == 2
        # answer tokens are supervised; some labels != -100
        assert (batch["labels"] != -100).any()
    else:
        for k in ("trace_ids", "trace_mask", "question_ids",
                  "question_mask", "answer_ids", "answer_mask"):
            assert batch[k].shape[0] == B and batch[k].ndim == 2


@pytest.mark.slow
def test_collator_eval_mode_has_no_answer(tok):
    cfg = ExperimentConfig(condition="z")
    coll = Collator(tok, cfg, "z", include_answer=False)
    batch = coll(_ROWS)
    assert "answer_ids" not in batch
    assert "question_ids" in batch and "trace_ids" in batch


@pytest.mark.slow
@pytest.mark.parametrize("condition", ["floor", "ceiling"])
def test_collator_eval_mode_is_left_padded(tok, condition):
    """Eval batches feed straight into batched `generate()`, which needs
    every row's last real token at the same rightmost column. The shorter
    row (index 0) must therefore have its padding at the start, not the end."""
    cfg = ExperimentConfig(condition=condition)
    coll = Collator(tok, cfg, condition, include_answer=False)
    batch = coll(_ROWS)
    mask = batch["attention_mask"]
    # row 0 (shorter prompt) has pad at the start: leading zero(s), then all ones
    row0 = mask[0].tolist()
    first_one = row0.index(1)
    assert all(v == 0 for v in row0[:first_one])
    assert all(v == 1 for v in row0[first_one:])
    # last column is real content for every row (generation continues from here)
    assert (mask[:, -1] == 1).all()


@pytest.mark.slow
def test_collator_z_eval_mode_question_is_left_padded(tok):
    cfg = ExperimentConfig(condition="z")
    coll = Collator(tok, cfg, "z", include_answer=False)
    batch = coll(_ROWS)
    qmask = batch["question_mask"]
    row0 = qmask[0].tolist()
    first_one = row0.index(1)
    assert all(v == 0 for v in row0[:first_one])
    assert all(v == 1 for v in row0[first_one:])
    assert (qmask[:, -1] == 1).all()


@pytest.mark.slow
def test_collator_training_mode_stays_right_padded(tok):
    """Training path must not change: teacher-forcing over the full
    sequence with -100-masked labels expects right-padding."""
    cfg = ExperimentConfig(condition="floor")
    coll = Collator(tok, cfg, "floor", include_answer=True)
    batch = coll(_ROWS)
    mask = batch["attention_mask"]
    row0 = mask[0].tolist()  # shorter row
    last_one = len(row0) - 1 - row0[::-1].index(1)
    assert all(v == 0 for v in row0[last_one + 1:])
    assert row0[0] == 1
