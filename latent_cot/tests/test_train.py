import pytest
from latent_cot.train import exact_match


def test_exact_match_normalizes():
    assert exact_match(["72", " 72 ", "The answer is 72."], ["72", "72", "72"]) == 1.0
    assert exact_match(["1,000"], ["1000"]) == 1.0
    assert exact_match(["8"], ["9"]) == 0.0


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
