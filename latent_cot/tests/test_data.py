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
