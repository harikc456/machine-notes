from __future__ import annotations
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from latent_cot.config import ExperimentConfig, load_config
from latent_cot.data import load_gsm8k, GSM8KDataset, Collator, normalize_number
from latent_cot.model import LatentCoTModel


def exact_match(preds: list[str], golds: list[str]) -> float:
    if not preds:
        return 0.0
    hits = sum(normalize_number(p) == g for p, g in zip(preds, golds))
    return hits / len(preds)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _subset(rows: list[dict], n: int) -> list[dict]:
    return rows if n <= 0 else rows[:n]


def train_and_eval(cfg: ExperimentConfig) -> dict:
    _seed_all(cfg.seed)
    model = LatentCoTModel(cfg)

    train_rows = _subset(load_gsm8k("train", cfg.strip_annotations), cfg.max_train_samples)
    eval_rows = _subset(load_gsm8k("test", cfg.strip_annotations), cfg.max_eval_samples)

    train_coll = Collator(model.tokenizer, cfg, cfg.condition, include_answer=True)
    eval_coll = Collator(model.tokenizer, cfg, cfg.condition, include_answer=False)

    train_loader = DataLoader(
        GSM8KDataset(train_rows), batch_size=cfg.batch_size, shuffle=True,
        collate_fn=train_coll,
    )
    eval_loader = DataLoader(
        GSM8KDataset(eval_rows), batch_size=cfg.batch_size, shuffle=False,
        collate_fn=eval_coll,
    )

    opt = AdamW(model.trainable_parameters(), lr=cfg.lr)
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    warmup = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 1.0 - prog)

    sched = LambdaLR(opt, lr_lambda)

    model.train()
    final_loss = float("nan")
    for _ in range(cfg.epochs):
        opt.zero_grad()
        for i, batch in enumerate(train_loader):
            loss = model(batch) / cfg.grad_accum_steps
            loss.backward()
            final_loss = loss.item() * cfg.grad_accum_steps
            if (i + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
                opt.zero_grad()

    # ---- eval ----
    model.eval()
    if cfg.condition == "reconstruct":
        correct, total, n_eval = 0, 0, 0
        with torch.no_grad():
            for batch in eval_loader:
                logits, labels = model.logits_and_labels(batch)
                preds_tok = logits[:, :-1, :].argmax(-1)
                targets = labels[:, 1:].to(preds_tok.device)
                mask = targets != -100
                correct += (preds_tok == targets)[mask].sum().item()
                total += mask.sum().item()
                n_eval += targets.size(0)
        return {
            "condition": cfg.condition,
            "token_accuracy": (correct / total) if total else 0.0,
            "n_eval": n_eval,
            "final_train_loss": final_loss,
        }

    preds, golds = [], []
    for batch in eval_loader:
        out = model.generate(batch, max_new_tokens=cfg.max_answer_tokens)
        preds.extend(out)
        golds.extend(batch["label_text"])

    return {
        "condition": cfg.condition,
        "eval_accuracy": exact_match(preds, golds),
        "n_eval": len(golds),
        "final_train_loss": final_loss,
    }


def _parse_args() -> ExperimentConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--condition", type=str, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else ExperimentConfig()
    if args.condition is not None:
        cfg.condition = args.condition
    if args.max_train_samples is not None:
        cfg.max_train_samples = args.max_train_samples
    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg.__post_init__()  # re-validate after overrides
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    result = train_and_eval(cfg)
    print(result)
