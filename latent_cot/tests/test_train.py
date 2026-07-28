import pytest
import torch
from latent_cot.train import exact_match, _token_accuracy_counts


def test_exact_match_normalizes():
    assert exact_match(["72", " 72 ", "The answer is 72."], ["72", "72", "72"]) == 1.0
    assert exact_match(["1,000"], ["1000"]) == 1.0
    assert exact_match(["8"], ["9"]) == 0.0


def test_token_accuracy_counts_shift_and_mask():
    # 2 timesteps of "prefix" (label=-100), 3 real target tokens per row
    # labels: [-100, -100, 5, 7, 9]  (row 0), [-100, -100, 5, 7, -100] (row 1, last pos padded)
    labels = torch.tensor([[-100, -100, 5, 7, 9], [-100, -100, 5, 7, -100]])
    logits = torch.zeros(2, 5, 10)
    # position t's logits predict labels[t+1], so:
    # row 0: logits[0,1] should argmax to 5 (predicts labels[0,2]), logits[0,2] argmax to 7,
    # logits[0,3] argmax to 9 -> all correct
    logits[0, 1, 5] = 10.0
    logits[0, 2, 7] = 10.0
    logits[0, 3, 9] = 10.0
    # row 1: logits[1,1] argmax to 5 (correct), logits[1,2] argmax to 0 (WRONG, should be 7)
    logits[1, 1, 5] = 10.0
    logits[1, 2, 0] = 10.0
    correct, total = _token_accuracy_counts(logits, labels)
    assert total == 5  # 3 real targets in row 0 + 2 real targets in row 1 (row 1's last position is -100, excluded)
    assert correct == 4  # all 3 in row 0, 1 of 2 in row 1


@pytest.mark.slow
def test_train_and_eval_smoke():
    from latent_cot.config import ExperimentConfig
    from latent_cot.train import train_and_eval
    cfg = ExperimentConfig(
        condition="z", n_slots=4, d_z=16, lora_r=4, epochs=1,
        batch_size=2, grad_accum_steps=1, max_train_samples=8, max_eval_samples=8,
        max_trace_tokens=64, max_question_tokens=32,
    )
    result = train_and_eval(cfg)
    assert set(result) >= {"condition", "eval_accuracy", "n_eval", "final_train_loss"}
    assert 0.0 <= result["eval_accuracy"] <= 1.0
    assert result["n_eval"] == 8


@pytest.mark.slow
def test_train_and_eval_reconstruct_smoke():
    from latent_cot.config import ExperimentConfig
    from latent_cot.train import train_and_eval
    cfg = ExperimentConfig(
        condition="reconstruct", n_slots=4, d_z=16, lora_r=4, diffusion_steps=2,
        epochs=1, batch_size=2, grad_accum_steps=1,
        max_train_samples=8, max_eval_samples=8,
        max_trace_tokens=64, max_question_tokens=32,
    )
    result = train_and_eval(cfg)
    assert set(result) >= {"condition", "token_accuracy", "n_eval", "final_train_loss"}
    assert 0.0 <= result["token_accuracy"] <= 1.0
    assert result["n_eval"] == 8
