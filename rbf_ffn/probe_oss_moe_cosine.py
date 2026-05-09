"""
Probe expert-pair cosine similarity in an open-source HuggingFace MoE model.

For each MoE block, records the mean pairwise cosine similarity among the top-k
active expert outputs per token. Reports per-layer statistics and an aggregate
histogram — directly comparable to probe_moe_cosine.py results on our own model.

Supported architectures (auto-detected):
  - OLMoE  (allenai/OLMoE-1B-7B)       — 64 experts, top_k=8
  - Mixtral (mistralai/Mixtral-8x7B-*)  — 8 experts,  top_k=2

Usage:
    # OLMoE (fits in ~14 GB bfloat16, uses device_map=auto for tight VRAM):
    python -m rbf_ffn.probe_oss_moe_cosine \
        --model allenai/OLMoE-1B-7B \
        --n_batches 50

    # Mixtral (needs ~24 GB; install bitsandbytes and add --load_in_4bit):
    python -m rbf_ffn.probe_oss_moe_cosine \
        --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --load_in_4bit \
        --n_batches 50
"""
from __future__ import annotations
import argparse
import types

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------

_MOE_BLOCK_NAMES = {
    "olmoesparsemoeblock",
    "mixtralsparsemoeblock",
}


def _is_moe_block(module: torch.nn.Module) -> bool:
    return type(module).__name__.lower() in _MOE_BLOCK_NAMES


# ---------------------------------------------------------------------------
# Generic probe forward
# ---------------------------------------------------------------------------

def _make_probe_forward(module: torch.nn.Module, store: dict):
    """
    Return a bound method that replaces module.forward.

    Runs only the top-k selected experts per token (same tokens each expert
    sees in the original forward), collects their outputs, computes all
    C(k,2) pairwise cosine similarities per token, and stores the mean.
    Then calls the original forward for the actual model output.
    """
    orig_forward = module.forward
    gate       = module.gate
    experts    = module.experts
    top_k      = module.top_k
    n_experts  = len(experts)

    def probe_forward(hidden_states: torch.Tensor, *args, **kwargs):
        B, N, D = hidden_states.shape
        flat = hidden_states.view(-1, D)   # (T, D)
        T    = flat.shape[0]

        with torch.no_grad():
            router_logits  = gate(flat)
            routing_probs  = F.softmax(router_logits.float(), dim=-1)
            _, selected    = torch.topk(routing_probs, top_k, dim=-1)  # (T, top_k)

            # collected[t, rank] = output of the rank-th selected expert for token t
            collected = torch.zeros(T, top_k, D, device=flat.device, dtype=flat.dtype)

            for e_idx in range(n_experts):
                # find which (token, rank) pairs use this expert
                mask = (selected == e_idx)          # (T, top_k) bool
                if not mask.any():
                    continue
                token_rows = mask.any(dim=1)        # (T,) — tokens that use expert e_idx
                e_out = experts[e_idx](flat[token_rows])  # (n_tok, D)

                # deposit output at the correct rank slot for each token
                for rank in range(top_k):
                    rank_mask = mask[:, rank]       # (T,) bool
                    if not rank_mask.any():
                        continue
                    # which positions in e_out correspond to rank_mask tokens?
                    # token_rows is the union; rank_mask is a subset
                    within = rank_mask[token_rows]  # (n_tok,) bool
                    collected[rank_mask, rank] = e_out[within]

            # pairwise cosine similarities among top-k expert outputs
            cos_sims = []
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    sim = F.cosine_similarity(
                        collected[:, i], collected[:, j], dim=-1
                    )  # (T,)
                    cos_sims.append(sim)

            mean_sim = torch.stack(cos_sims, dim=0).mean(dim=0)  # (T,)
            store["cos_sims"].append(mean_sim.cpu())

        return orig_forward(hidden_states, *args, **kwargs)

    return probe_forward


def install_probes(model: torch.nn.Module) -> list[dict]:
    stores = []
    for name, module in model.named_modules():
        if not _is_moe_block(module):
            continue
        store = {"name": name, "cos_sims": []}
        stores.append(store)
        probe_fwd = _make_probe_forward(module, store)
        module.forward = types.MethodType(
            lambda self, hs, *a, _fwd=probe_fwd, **kw: _fwd(hs, *a, **kw),
            module,
        )
    return stores


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(stores: list[dict], model_name: str) -> None:
    print(f"\n=== Expert cosine similarity: {model_name} ===\n")
    print(f"{'Layer':<55} {'Mean':>8} {'Std':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8}")
    print("-" * 103)

    all_sims = []
    for st in stores:
        if not st["cos_sims"]:
            continue
        sims = torch.cat(st["cos_sims"])
        all_sims.append(sims)
        p = torch.quantile(sims, torch.tensor([0.25, 0.50, 0.75, 0.95]))
        name = st["name"][-54:] if len(st["name"]) > 54 else st["name"]
        print(
            f"{name:<55}"
            f" {sims.mean().item():>8.4f}"
            f" {sims.std().item():>8.4f}"
            f" {p[0].item():>8.4f}"
            f" {p[1].item():>8.4f}"
            f" {p[2].item():>8.4f}"
            f" {p[3].item():>8.4f}"
        )

    if not all_sims:
        print("No data collected — no MoE blocks found.")
        return

    print("-" * 103)
    agg = torch.cat(all_sims)
    p   = torch.quantile(agg, torch.tensor([0.25, 0.50, 0.75, 0.95]))
    print(
        f"{'ALL LAYERS':<55}"
        f" {agg.mean().item():>8.4f}"
        f" {agg.std().item():>8.4f}"
        f" {p[0].item():>8.4f}"
        f" {p[1].item():>8.4f}"
        f" {p[2].item():>8.4f}"
        f" {p[3].item():>8.4f}"
    )

    print("\nCosine similarity distribution (all layers, mean pairwise per token):")
    bins   = torch.linspace(-1.0, 1.0, 21)
    counts = torch.histc(agg, bins=20, min=-1.0, max=1.0)
    total  = counts.sum().item()
    for i, c in enumerate(counts):
        lo, hi = bins[i].item(), bins[i + 1].item()
        bar = "#" * int(50 * c.item() / total)
        print(f"  [{lo:+.2f}, {hi:+.2f})  {bar}  {c.item():.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B",
                        help="HuggingFace model ID")
    parser.add_argument("--n_batches", type=int, default=50,
                        help="Number of text chunks to probe (default: 50)")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load in 4-bit (requires bitsandbytes)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    load_kwargs: dict = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    print(f"Loading model: {args.model}  (device_map=auto, bfloat16)")
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    stores = install_probes(model)
    if not stores:
        print("ERROR: No MoE blocks detected. Check model architecture.")
        return
    print(f"Probing {len(stores)} MoE layer(s)...\n")

    # WikiText-103 test split — same domain as our training data
    print("Loading WikiText-103 test split...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="test")
    text    = "\n\n".join(dataset["text"])
    tokens  = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

    chunk_size = args.seq_len
    n_chunks   = min(args.n_batches, len(tokens) // chunk_size)
    print(f"Running {n_chunks} chunks of {chunk_size} tokens each...\n")

    with torch.no_grad():
        for i in range(n_chunks):
            chunk = tokens[i * chunk_size : (i + 1) * chunk_size].unsqueeze(0)
            chunk = chunk.to(next(model.parameters()).device)
            model(chunk)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n_chunks} chunks done")

    report(stores, args.model)


if __name__ == "__main__":
    main()
