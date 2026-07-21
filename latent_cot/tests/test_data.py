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
from latent_cot.data import GSM8KDataset, Collator

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
