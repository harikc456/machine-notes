"""
Probe expert-pair cosine similarity in a trained MoE checkpoint.

Loads a checkpoint, runs N val batches through the model, and for each MoE
layer records the cosine similarity between the top-2 active experts' outputs
per token. Reports mean, std, and percentile distribution per layer.

Usage:
    python -m rbf_ffn.probe_moe_cosine \
        --exp rbf_ffn/experiments/20260508_085651_652695_xsa_moe_qknorm_wnorm_d256 \
        --n_batches 50
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from rbf_ffn.config import load_config
from rbf_ffn.data import get_dataloaders
from rbf_ffn.models.model import CausalLM
from rbf_ffn.models.moe_ffn import SparseMoEFFN


def _install_probes(model: CausalLM) -> list[dict]:
    """
    Replace each SparseMoEFFN.forward with a probe version that computes
    cosine similarity between the two active expert outputs per token and
    accumulates results in a shared store.

    Returns a list of per-layer stores (one dict per MoE layer).
    """
    stores = []

    for name, module in model.named_modules():
        if not isinstance(module, SparseMoEFFN):
            continue

        store = {"cos_sims": [], "name": name}
        stores.append(store)

        # capture module and store in closure
        def make_probe_forward(mod, st):
            orig_forward = mod.forward

            def probe_forward(x: torch.Tensor) -> torch.Tensor:
                B, N, D = x.shape
                x_flat = x.view(B * N, D)

                logits  = mod.router(x_flat)
                weights = F.softmax(logits, dim=-1)
                top_w, top_idx = torch.topk(weights, mod.top_k, dim=-1)
                top_w = top_w / top_w.sum(dim=-1, keepdim=True)

                # compute all expert outputs for all tokens (probe only)
                with torch.no_grad():
                    all_outputs = torch.stack(
                        [e(x_flat) for e in mod.experts], dim=1
                    )  # (T, n_experts, D)

                T = B * N
                e0 = all_outputs[torch.arange(T), top_idx[:, 0]]  # (T, D)
                e1 = all_outputs[torch.arange(T), top_idx[:, 1]]  # (T, D)

                cos_sim = F.cosine_similarity(e0, e1, dim=-1)  # (T,)
                st["cos_sims"].append(cos_sim.detach().cpu())

                # use original forward output (routing mask path) for correctness
                return orig_forward(x)

            return probe_forward

        module.forward = make_probe_forward(module, store)

    return stores


def report(stores: list[dict]) -> None:
    print(f"\n{'Layer':<40} {'Mean':>8} {'Std':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
    print("-" * 88)

    all_sims = []
    for st in stores:
        sims = torch.cat(st["cos_sims"])  # (total_tokens,)
        all_sims.append(sims)
        p = torch.quantile(sims, torch.tensor([0.25, 0.50, 0.75, 0.95]))
        print(
            f"{st['name']:<40}"
            f" {sims.mean().item():>8.4f}"
            f" {sims.std().item():>8.4f}"
            f" {p[0].item():>8.4f}"
            f" {p[1].item():>8.4f}"
            f" {p[2].item():>8.4f}"
            f" {p[3].item():>8.4f}"
        )

    print("-" * 88)
    agg = torch.cat(all_sims)
    p = torch.quantile(agg, torch.tensor([0.25, 0.50, 0.75, 0.95]))
    print(
        f"{'ALL LAYERS':<40}"
        f" {agg.mean().item():>8.4f}"
        f" {agg.std().item():>8.4f}"
        f" {p[0].item():>8.4f}"
        f" {p[1].item():>8.4f}"
        f" {p[2].item():>8.4f}"
        f" {p[3].item():>8.4f}"
    )

    # histogram of cosine similarity
    print("\nCosine similarity distribution (all layers):")
    bins = torch.linspace(-1.0, 1.0, 21)
    counts = torch.histc(agg, bins=20, min=-1.0, max=1.0)
    total = counts.sum().item()
    for i, c in enumerate(counts):
        lo, hi = bins[i].item(), bins[i + 1].item()
        bar = "#" * int(50 * c.item() / total)
        print(f"  [{lo:+.2f}, {hi:+.2f})  {bar}  {c.item():.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", required=True,
        help="Path to experiment directory (must contain config.yaml and checkpoint_best.pt)"
    )
    parser.add_argument("--n_batches", type=int, default=50,
                        help="Number of val batches to probe (default: 50)")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt",
                        help="Checkpoint filename within the experiment dir")
    args = parser.parse_args()

    exp_dir = Path(args.exp)
    config_path = exp_dir / "config.yaml"
    ckpt_path   = exp_dir / args.checkpoint

    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")

    model = CausalLM(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    stores = _install_probes(model)
    if not stores:
        print("No SparseMoEFFN layers found in model.")
        return
    print(f"Probing {len(stores)} MoE layer(s) over {args.n_batches} val batches...\n")

    _, val_loader, _ = get_dataloaders(cfg)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_batches:
                break
            x = batch[:, :-1].to(device)
            model(x)

    report(stores)


if __name__ == "__main__":
    main()
