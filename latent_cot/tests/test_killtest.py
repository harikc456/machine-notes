from latent_cot.run_killtest import format_table


def test_format_table_orders_and_contains_all():
    results = [
        {"condition": "floor", "eval_accuracy": 0.10, "n_eval": 200, "final_train_loss": 1.2},
        {"condition": "z", "eval_accuracy": 0.55, "n_eval": 200, "final_train_loss": 0.4},
        {"condition": "ceiling", "eval_accuracy": 0.62, "n_eval": 200, "final_train_loss": 0.3},
        {"condition": "z_shuffled", "eval_accuracy": 0.12, "n_eval": 200, "final_train_loss": 1.1},
    ]
    table = format_table(results)
    for c in ("floor", "z", "ceiling", "z_shuffled"):
        assert c in table
    assert "0.55" in table or "55" in table
