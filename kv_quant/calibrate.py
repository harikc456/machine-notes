"""SpectralQuant calibration: per-head eigenvectors and Lloyd-Max codebooks.

Usage:
    python -m kv_quant.calibrate \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --output spectralquant_qwen25_7b \\
        --n-seqs 100 --bits 4
"""
from __future__ import annotations
import argparse
import os
import sys
import torch
from tqdm import tqdm

_SPECTRALQUANT_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "spectralquant", "src")
)
if _SPECTRALQUANT_SRC not in sys.path:
    sys.path.insert(0, _SPECTRALQUANT_SRC)

from spectralquant.calibration import (
    EigenspectralCalibrator,
    HeadCalibrationData,
    _compute_covariance,
    _eigendecompose,
    _participation_ratio,
    _spectral_gap,
    _cumulative_variance_thresholds,
)
from spectralquant.nonuniform_quantization import NonUniformQuantizer


def _calibrate_from_kv(
    all_keys: list[list[torch.Tensor]],
    all_vals: list[list[torch.Tensor]],
    head_dim: int,
    bits: int,
    base_path: str,
) -> None:
    """Compute calibration from pre-collected K/V tensors and save results.

    Parameters
    ----------
    all_keys:
        all_keys[layer_idx] = list of (n_kv_heads, seq_len, head_dim) float tensors.
    all_vals:
        Same shape as all_keys but for values.
    head_dim:
        Head dimension.
    bits:
        Target average bits per coordinate (passed as avg_bits to NonUniformQuantizer).
    base_path:
        Output base path without extension. Writes <base>.pt, <base>_meta.json,
        <base>_quantizers.pt.
    """
    n_layers = len(all_keys)
    n_kv_heads = all_keys[0][0].shape[0] if all_keys and all_keys[0] else 0

    calibrator = EigenspectralCalibrator()
    quant_state: dict = {}

    for layer_idx in range(n_layers):
        if not all_keys[layer_idx]:
            continue
        # (n_kv_heads, total_tokens, head_dim)
        layer_keys = torch.cat(all_keys[layer_idx], dim=1).float()
        layer_vals = torch.cat(all_vals[layer_idx], dim=1).float()

        for head_idx in range(n_kv_heads):
            for kv_type, layer_data in (("key", layer_keys), ("value", layer_vals)):
                vectors = layer_data[head_idx]  # (total_tokens, head_dim)
                if vectors.shape[0] < 2:
                    continue

                cov = _compute_covariance(vectors)
                eigenvalues, eigenvectors = _eigendecompose(cov)
                d_eff = _participation_ratio(eigenvalues)
                gap = _spectral_gap(eigenvalues, d_eff)
                var_95, var_99 = _cumulative_variance_thresholds(eigenvalues)

                calibrator._calibration_data[(layer_idx, head_idx, kv_type)] = HeadCalibrationData(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    head_type=kv_type,
                    eigenvalues=eigenvalues,
                    eigenvectors=eigenvectors,
                    d_eff=d_eff,
                    spectral_gap=gap,
                    var_95=var_95,
                    var_99=var_99,
                    n_samples=vectors.shape[0],
                    head_dim=head_dim,
                )

                # Forward rotation: x @ V (same as SpectralRotation.rotate)
                rotated = vectors @ eigenvectors  # (n_tokens, head_dim)
                quant = NonUniformQuantizer(eigenvalues=eigenvalues, avg_bits=float(bits))
                quant.fit(rotated, d_eff=d_eff)

                quant_state[f"L{layer_idx}_H{head_idx}_{kv_type}"] = {
                    "semantic_centroids": quant._semantic_quantizer._centroids.clone(),
                    "tail_centroids": quant._tail_quantizer._centroids.clone(),
                    "d_eff_int": quant._d_eff_int,
                    "b_high": quant._b_high,
                    "b_low": quant._b_low,
                    "head_dim": head_dim,
                }

    calibrator._is_calibrated = True
    calibrator.save(base_path)
    torch.save(quant_state, base_path + "_quantizers.pt")
    print(f"Saved: {base_path}.pt / {base_path}_meta.json / {base_path}_quantizers.pt")


def calibrate(
    model_id: str,
    base_path: str,
    n_seqs: int = 100,
    bits: int = 4,
    device: str = "cuda",
) -> None:
    """Calibrate a model and save results to disk.

    Parameters
    ----------
    model_id:
        HuggingFace model ID.
    base_path:
        Output base path without extension.
    n_seqs:
        Number of wikitext sequences to process.
    bits:
        Target average bits per coordinate.
    device:
        Torch device string.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    texts = [ex["text"] for ex in dataset if len(ex["text"].strip()) > 100][:n_seqs]

    # Some models (e.g. Gemma 4) nest arch params under text_config
    arch_cfg = getattr(model.config, "text_config", model.config)
    n_layers = arch_cfg.num_hidden_layers
    n_kv_heads = getattr(arch_cfg, "num_key_value_heads", arch_cfg.num_attention_heads)
    head_dim = getattr(
        arch_cfg, "head_dim",
        arch_cfg.hidden_size // arch_cfg.num_attention_heads,
    )

    all_keys: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]
    all_vals: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    for text in tqdm(texts, desc="Collecting KV vectors"):
        ids = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).input_ids.to(device)
        cache = DynamicCache()
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)
        for l in range(min(n_layers, len(cache.key_cache))):
            all_keys[l].append(cache.key_cache[l][0].float().cpu())
            all_vals[l].append(cache.value_cache[l][0].float().cpu())

    _calibrate_from_kv(all_keys, all_vals, head_dim=head_dim, bits=bits, base_path=base_path)
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="Base path without extension")
    parser.add_argument("--n-seqs", type=int, default=100)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    calibrate(args.model, args.output, args.n_seqs, args.bits, args.device)
