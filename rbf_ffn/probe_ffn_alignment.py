"""
Probe per-layer FFN output alignment with its input (norm2(x)).

Measures cosine similarity between y = mlp(norm2(x)) and norm2(x) for each
transformer block. High alignment means the FFN is naturally writing a
component along the residual stream direction — i.e., OrthogonalMLPWrapper is
doing real work in that layer. Low alignment means the FFN already writes
perpendicular updates and the wrapper is nearly a no-op.

Works on two checkpoint types:
  - With OrthogonalMLPWrapper: intercepts y before the parallel projection.
  - Without (plain FFN): hooks the FFN forward directly.

Usage:
    python -m rbf_ffn.probe_ffn_alignment \\
        --exp rbf_ffn/experiments/20260416_105724_190084_xsa_swiglu_qknorm_wnorm_orthogonal_d256 \\
        --n_batches 50

    python -m rbf_ffn.probe_ffn_alignment \\
        --exp rbf_ffn/experiments/20260430_132442_213125_xsa_swiglu_qknorm_wnorm_d256 \\
        --n_batches 50
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from rbf_ffn.config import load_config
from rbf_ffn.data import get_dataloaders
from rbf_ffn.models.model import CausalLM
from rbf_ffn.models.orthogonal_ffn import OrthogonalMLPWrapper, GatedOrthogonalMLPWrapper
from rbf_ffn.models.transformer_block import TransformerBlock


def _install_probes(model: CausalLM) -> list[dict]:
    """
    For each TransformerBlock, hook the FFN forward to capture (y, x) pairs and
    accumulate per-token cosine similarity between the raw FFN output y and the
    FFN input x (= norm2(residual)).

    Returns a list of per-layer stores ordered by block index.
    """
    stores: list[dict] = []

    for name, module in model.named_modules():
        if not isinstance(module, TransformerBlock):
            continue

        store = {"cos_sims": [], "name": name, "wrapped": False}
        stores.append(store)
        ffn = module.ffn

        if isinstance(ffn, (OrthogonalMLPWrapper, GatedOrthogonalMLPWrapper)):
            # Intercept inside the wrapper: capture y = inner_mlp(x) before projection.
            store["wrapped"] = True
            inner_mlp = ffn.mlp
            orig_inner = inner_mlp.forward
            eps = ffn.eps

            def make_wrapped_probe(st, orig, wrapper_eps):
                def probe_inner(x: torch.Tensor) -> torch.Tensor:
                    y = orig(x)
                    cos = F.cosine_similarity(y.detach(), x.detach(), dim=-1)  # (B, N)
                    st["cos_sims"].append(cos.reshape(-1).cpu())
                    # also record the scalar projection magnitude
                    dot_yx = (y.detach() * x.detach()).sum(-1)
                    dot_xx = (x.detach() * x.detach()).sum(-1) + wrapper_eps
                    st.setdefault("proj_mag", []).append(
                        (dot_yx / dot_xx).abs().reshape(-1).cpu()
                    )
                    return y
                return probe_inner

            inner_mlp.forward = make_wrapped_probe(store, orig_inner, eps)

        else:
            # Plain FFN: hook module.ffn.forward directly.
            orig_ffn = ffn.forward

            def make_plain_probe(st, orig):
                def probe_ffn(x: torch.Tensor) -> torch.Tensor:
                    y = orig(x)
                    cos = F.cosine_similarity(y.detach(), x.detach(), dim=-1)
                    st["cos_sims"].append(cos.reshape(-1).cpu())
                    dot_yx = (y.detach() * x.detach()).sum(-1)
                    dot_xx = (x.detach() * x.detach()).sum(-1) + 1e-8
                    st.setdefault("proj_mag", []).append(
                        (dot_yx / dot_xx).abs().reshape(-1).cpu()
                    )
                    return y
                return probe_ffn

            ffn.forward = make_plain_probe(store, orig_ffn)

    return stores


def report(stores: list[dict]) -> None:
    print(
        f"\n{'Layer':<50} {'Wrapped':>7} {'Mean cos':>9} {'Std':>8} "
        f"{'p25':>8} {'p75':>8} {'p95':>8} {'Mean|proj|':>11}"
    )
    print("-" * 115)

    all_sims: list[torch.Tensor] = []
    for st in stores:
        sims = torch.cat(st["cos_sims"])
        mags = torch.cat(st["proj_mag"]) if "proj_mag" in st else torch.zeros_like(sims)
        all_sims.append(sims)
        p = torch.quantile(sims, torch.tensor([0.25, 0.75, 0.95]))
        print(
            f"{st['name']:<50}"
            f" {'yes' if st['wrapped'] else 'no':>7}"
            f" {sims.mean().item():>9.4f}"
            f" {sims.std().item():>8.4f}"
            f" {p[0].item():>8.4f}"
            f" {p[1].item():>8.4f}"
            f" {p[2].item():>8.4f}"
            f" {mags.mean().item():>11.4f}"
        )

    print("-" * 115)
    agg = torch.cat(all_sims)
    p = torch.quantile(agg, torch.tensor([0.25, 0.75, 0.95]))
    print(
        f"{'ALL LAYERS':<50}"
        f" {'—':>7}"
        f" {agg.mean().item():>9.4f}"
        f" {agg.std().item():>8.4f}"
        f" {p[0].item():>8.4f}"
        f" {p[1].item():>8.4f}"
        f" {p[2].item():>8.4f}"
    )

    # rank layers by mean cosine similarity
    print("\nLayer ranking (highest alignment first — best candidates to keep orthogonal_ffn on):")
    ranked = sorted(stores, key=lambda s: torch.cat(s["cos_sims"]).mean().item(), reverse=True)
    for rank, st in enumerate(ranked, 1):
        mean_cos = torch.cat(st["cos_sims"]).mean().item()
        print(f"  {rank}. {st['name']:<48}  mean cos = {mean_cos:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp", required=True,
        help="Path to experiment directory (must contain config.yaml and a checkpoint)"
    )
    parser.add_argument("--n_batches", type=int, default=50,
                        help="Number of val batches to probe (default: 50)")
    parser.add_argument("--checkpoint", default="checkpoint_best.pt",
                        help="Checkpoint filename within the experiment dir")
    args = parser.parse_args()

    exp_dir = Path(args.exp)
    cfg = load_config(exp_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {exp_dir / args.checkpoint}")

    model = CausalLM(cfg).to(device)
    state = torch.load(exp_dir / args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    stores = _install_probes(model)
    print(
        f"Probing {len(stores)} block(s) "
        f"({sum(s['wrapped'] for s in stores)} with OrthogonalMLPWrapper) "
        f"over {args.n_batches} val batches...\n"
    )

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
