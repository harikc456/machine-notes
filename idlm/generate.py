# idlm/generate.py
"""
Introspective Strided Decoding (ISD) for I-DLM.

Usage:
    python -m idlm.generate --config idlm/configs/baseline.yaml \
        --checkpoint idlm/experiments/<run>/checkpoint_best.pt \
        --output results.jsonl
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from idlm.config import IDLMConfig, load_config

MASK_TOKEN_ID = 50256


def compute_tpf_oh(alpha: float, stride: int) -> float:
    """
    Tokens Per Forward / Overhead ratio.
    TPF/OH = stride / (1 + (1 - alpha) * stride)
    """
    return stride / (1.0 + (1.0 - alpha) * stride)


@torch.no_grad()
def isd_generate(
    model,
    prompt: list[int],
    cfg: IDLMConfig,
    device: torch.device,
) -> list[int]:
    """
    Generate cfg.gen_len tokens after the prompt using ISD.
    Samples from q distribution (LoRA active) at each stride window.
    Returns the full sequence (prompt + generated tokens).
    """
    model.eval()
    accepted = list(prompt)
    gen_len = cfg.gen_len
    stride = cfg.stride
    vocab_size = cfg.vocab_size
    mask_id = min(MASK_TOKEN_ID, vocab_size - 1)

    while len(accepted) - len(prompt) < gen_len:
        remaining = gen_len - (len(accepted) - len(prompt))
        s = min(stride, remaining)
        current_len = len(accepted)

        tokens = torch.tensor(
            accepted + [mask_id] * s, dtype=torch.long, device=device
        ).unsqueeze(0)                                        # (1, current_len + s)

        total_len = tokens.size(1)
        use_lora_mask = torch.zeros(1, total_len, 1, device=device)
        use_lora_mask[:, current_len:, :] = 1.0

        logits = model(tokens, use_lora_mask)                 # (1, total_len, V)
        q_logits = logits[0, current_len:, :]                 # (s, V)
        q_probs = F.softmax(q_logits, dim=-1)
        proposed = torch.multinomial(q_probs, num_samples=1).squeeze(-1).tolist()
        accepted.extend(proposed)

    return accepted[:len(prompt) + gen_len]


@torch.no_grad()
def isd_acceptance_rate(
    model,
    sequence: list[int],
    cfg: IDLMConfig,
    device: torch.device,
) -> float:
    """
    Compute the introspective acceptance rate α on a complete sequence.

    Slides stride-S windows over the generated portion. For each window:
    - q forward: LoRA active at window positions (decode)
    - p forward: LoRA inactive everywhere (base AR)
    - α_step = mean_k min(1, p(x_k) / q(x_k))

    Returns mean α across all windows.
    """
    model.eval()
    prompt_len = cfg.prompt_len
    stride = cfg.stride
    gen_len = cfg.gen_len
    mask_id = min(MASK_TOKEN_ID, cfg.vocab_size - 1)

    gen_tokens = sequence[prompt_len:prompt_len + gen_len]
    if len(gen_tokens) < stride:
        return 1.0

    alphas = []
    for start in range(0, len(gen_tokens) - stride + 1, stride):
        window = gen_tokens[start:start + stride]
        prefix = sequence[:prompt_len + start]
        prefix_len = len(prefix)

        # q forward: LoRA active at MASK positions
        input_q = torch.tensor(
            prefix + [mask_id] * stride, dtype=torch.long, device=device
        ).unsqueeze(0)
        total_len = input_q.size(1)
        lora_mask = torch.zeros(1, total_len, 1, device=device)
        lora_mask[:, prefix_len:, :] = 1.0
        q_logits = model(input_q, lora_mask)[0, prefix_len:, :]   # (stride, V)
        q_probs = F.softmax(q_logits, dim=-1)

        # p forward: LoRA inactive everywhere (base AR weights)
        input_p = torch.tensor(
            prefix + window, dtype=torch.long, device=device
        ).unsqueeze(0)
        lora_off = torch.zeros(1, total_len, 1, device=device)
        p_logits = model(input_p, lora_off)[0, prefix_len:, :]     # (stride, V)
        p_probs = F.softmax(p_logits, dim=-1)

        window_t = torch.tensor(window, dtype=torch.long, device=device)
        q_sel = q_probs[torch.arange(stride), window_t]            # (stride,)
        p_sel = p_probs[torch.arange(stride), window_t]            # (stride,)
        ratios = torch.minimum(
            torch.ones(stride, device=device),
            p_sel / (q_sel + 1e-10),
        )
        alphas.append(ratios.mean().item())

    return sum(alphas) / len(alphas) if alphas else 1.0


def evaluate_isd(
    model,
    test_loader,
    cfg: IDLMConfig,
    device: torch.device,
    output_path: Path,
) -> dict:
    """Run full ISD evaluation on cfg.num_eval_examples test sequences."""
    results = []
    count = 0

    for batch in tqdm(test_loader, desc="ISD eval"):
        batch = batch.to(device)
        for i in range(batch.size(0)):
            if count >= cfg.num_eval_examples:
                break
            seq = batch[i].tolist()
            prompt = seq[:cfg.prompt_len]
            reference = seq[cfg.prompt_len:cfg.prompt_len + cfg.gen_len]

            generated = isd_generate(model, prompt, cfg, device)
            alpha = isd_acceptance_rate(model, generated, cfg, device)

            ref_tensor = torch.tensor(
                prompt + reference, dtype=torch.long, device=device
            ).unsqueeze(0)
            lora_off = torch.zeros(1, len(prompt) + len(reference), 1, device=device)
            logits = model(ref_tensor, lora_off)
            ppl_loss = F.cross_entropy(
                logits[0, cfg.prompt_len - 1:-1],
                ref_tensor[0, cfg.prompt_len:],
            )
            ppl = math.exp(ppl_loss.item())
            tpf_oh = compute_tpf_oh(alpha, cfg.stride)

            results.append({"alpha": alpha, "ppl": ppl, "tpf_oh": tpf_oh,
                             "prompt_ids": prompt, "generated_ids": generated})
            count += 1

        if count >= cfg.num_eval_examples:
            break

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    if not results:
        return {"alpha_mean": 0.0, "ppl_mean": float("inf"), "tpf_oh_mean": 0.0, "n_examples": 0}

    alpha_mean = sum(r["alpha"]  for r in results) / len(results)
    ppl_mean   = sum(r["ppl"]   for r in results) / len(results)
    tpf_mean   = sum(r["tpf_oh"] for r in results) / len(results)
    summary = {"alpha_mean": alpha_mean, "ppl_mean": ppl_mean,
               "tpf_oh_mean": tpf_mean, "n_examples": len(results)}
    print(f"ISD summary: {summary}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output",     default="idlm_isd_results.jsonl")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from rbf_ffn.config import load_config as load_ar_config
    from rbf_ffn.models.model import CausalLM
    from idlm.models.idlm_model import IDLMCausalLM

    ckpt_path = Path(args.checkpoint)
    ar_config_yaml = Path(cfg.ar_checkpoint).parent / "config.yaml"
    ar_cfg = load_ar_config(ar_config_yaml)
    ar_model = CausalLM(ar_cfg).to(device)

    ar_ckpt = torch.load(cfg.ar_checkpoint, map_location=device, weights_only=True)
    ar_model.load_state_dict(ar_ckpt["model"])

    model = IDLMCausalLM(ar_model, cfg.lora_rank, cfg.lora_alpha, cfg.lora_target_modules)
    lora_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(lora_ckpt["lora_state"], strict=False)
    print(f"Loaded LoRA checkpoint: {len(missing)} missing, {len(unexpected)} unexpected keys")

    from idlm.data import get_dataloaders
    _, _, test_loader = get_dataloaders(cfg)
    evaluate_isd(model, test_loader, cfg, device, Path(args.output))
