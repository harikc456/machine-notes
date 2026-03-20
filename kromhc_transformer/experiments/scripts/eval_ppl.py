"""Evaluate perplexity of a saved checkpoint on train and val splits."""
from __future__ import annotations
import argparse
import math
import torch
import torch.nn.functional as F
from pathlib import Path

from kromhc_transformer.config import KromHCConfig
from kromhc_transformer.data import get_dataloaders
from kromhc_transformer.models.model import CausalLM


@torch.no_grad()
def evaluate(model: CausalLM, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum, token_count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        inputs, targets = batch[:, :-1], batch[:, 1:]
        use_cuda = device.type == "cuda"
        with torch.autocast("cuda" if use_cuda else "cpu", dtype=torch.bfloat16, enabled=True):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss_sum += loss.item() * inputs.numel()
        token_count += inputs.numel()
    avg_loss = loss_sum / token_count
    return avg_loss, math.exp(avg_loss)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model_*.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Build config matching the saved run
    cfg = KromHCConfig(
        d_model=256, n_heads=8, n_layers=6, dropout=0.1,
        model_type="kromhc", use_kromhc=True, qk_norm=True, ffn_hidden=688,
        seq_len=512, vocab_size=50257, seed=42, batch_size=16,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    model = CausalLM(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)

    train_loader, val_loader, _ = get_dataloaders(cfg)

    print("Evaluating on train split...")
    train_loss, train_ppl = evaluate(model, train_loader, device)
    print(f"  train_loss={train_loss:.4f}  train_ppl={train_ppl:.2f}")

    print("Evaluating on val split...")
    val_loss, val_ppl = evaluate(model, val_loader, device)
    print(f"  val_loss={val_loss:.4f}    val_ppl={val_ppl:.2f}")


if __name__ == "__main__":
    main()
