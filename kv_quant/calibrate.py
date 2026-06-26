from __future__ import annotations
"""SpectralQuant calibration: compute per-head eigenvectors and VQ codebooks.

Usage:
    python -m kv_quant.calibrate \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output spectralquant_qwen25_7b.pt \
        --n-seqs 100 --bits 4
"""
import argparse
import torch
import numpy as np
from tqdm import tqdm

from kv_quant.ops.qjl import make_sign_matrix


def _compute_bit_split(
    total_bits: int, d: int, d_s: int, signal_bit_boost: float, max_bits: int = 8
) -> tuple[int, int]:
    """Allocate bits to signal and noise dims.

    Solves: (d_s * bits_s + (d - d_s) * bits_n) / d ≈ total_bits
    with bits_s = min(max_bits, round(total_bits * signal_bit_boost)).
    """
    bits_s = min(max_bits, round(total_bits * signal_bit_boost))
    d_noise = d - d_s
    if d_noise > 0:
        bits_n_float = (d * total_bits - d_s * bits_s) / d_noise
        bits_n = max(1, round(bits_n_float))
    else:
        bits_n = 1
    return bits_s, bits_n


def _kmeans_codebook(data: np.ndarray, n_centroids: int) -> np.ndarray:
    """Train Lloyd-Max codebook via k-means. Returns (n_centroids, k) float32."""
    from sklearn.cluster import KMeans
    n_centroids = min(n_centroids, len(data))
    km = KMeans(n_clusters=n_centroids, n_init=10, random_state=42, max_iter=300)
    km.fit(data)
    return km.cluster_centers_.astype(np.float32)


def calibrate(
    model_id: str,
    output_path: str,
    n_seqs: int = 100,
    bits: int = 4,
    signal_bit_boost: float = 2.0,
    qjl_dim: int = 32,
    device: str = "cuda",
) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    ).eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [ex["text"] for ex in dataset if len(ex["text"].strip()) > 100][:n_seqs]

    n_layers = model.config.num_hidden_layers
    n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    head_dim = getattr(
        model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads
    )

    # Collect key vectors: all_keys[layer] = list of (kv_heads, seq, d) cpu tensors
    all_keys: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    for text in tqdm(texts, desc="Collecting key vectors"):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).input_ids
        ids = ids.to(device)
        cache = DynamicCache()
        with torch.no_grad():
            model(ids, past_key_values=cache, use_cache=True)
        for l in range(min(n_layers, len(cache.key_cache))):
            # key_cache[l]: (1, kv_heads, seq, d) — squeeze batch dim
            all_keys[l].append(cache.key_cache[l][0].float().cpu())

    cal_data: dict = {
        "model_id": model_id,
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "qjl_dim": qjl_dim,
        "layers": {},
    }

    for layer_idx in tqdm(range(n_layers), desc="Computing per-head calibration"):
        if not all_keys[layer_idx]:
            continue
        # Stack along seq dim: (kv_heads, total_tokens, d)
        layer_keys = torch.cat(all_keys[layer_idx], dim=1)
        cal_data["layers"][layer_idx] = {}

        for head_idx in range(n_kv_heads):
            keys = layer_keys[head_idx]  # (total_tokens, d)

            # Covariance (centered)
            keys_c = keys - keys.mean(dim=0)
            cov = (keys_c.T @ keys_c) / max(keys_c.shape[0] - 1, 1)

            # Eigen-decomposition (ascending from eigh → flip to descending)
            eigenvalues, U = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.flip(0)
            U = U.flip(1)  # (d, d), columns = eigenvectors in descending order

            # Effective dimensionality
            d_eff = (eigenvalues.sum() ** 2) / ((eigenvalues ** 2).sum() + 1e-12)
            d_s = int(max(1, min(d_eff.ceil().item(), head_dim - 1)))

            bits_s, bits_n = _compute_bit_split(bits, head_dim, d_s, signal_bit_boost)

            # Project calibration data
            h_proj = keys @ U  # (total_tokens, d)
            h_sig = h_proj[:, :d_s].numpy()         # (total_tokens, d_s)
            h_noi = h_proj[:, d_s:].numpy()         # (total_tokens, d-d_s)

            # Codebooks
            cb_sig = _kmeans_codebook(h_sig, 2 ** bits_s)
            cb_noi = _kmeans_codebook(h_noi, 2 ** bits_n) if h_noi.shape[1] > 0 else np.zeros((1, 0), dtype=np.float32)

            S_signal = make_sign_matrix(qjl_dim, d_s)

            cal_data["layers"][layer_idx][head_idx] = {
                "U": U,                                          # (d, d)
                "d_s": d_s,
                "bits_signal": bits_s,
                "bits_noise": bits_n,
                "codebook_signal": torch.from_numpy(cb_sig),    # (2^bits_s, d_s)
                "codebook_noise": torch.from_numpy(cb_noi),     # (2^bits_n, d-d_s)
                "S_signal": S_signal,                           # (m, d_s)
            }

    # Compute representative top-level bit allocations (max across all heads/layers)
    all_bits_s = [head["bits_signal"] for layer_heads in cal_data["layers"].values() for head in layer_heads.values()]
    all_bits_n = [head["bits_noise"]  for layer_heads in cal_data["layers"].values() for head in layer_heads.values()]
    bits_signal_top = int(max(all_bits_s))
    bits_noise_top  = int(min(all_bits_n))

    # Add top-level bits to calibration data
    cal_data["bits_signal"] = bits_signal_top
    cal_data["bits_noise"] = bits_noise_top

    torch.save(cal_data, output_path)
    print(f"Calibration data saved to {output_path}")
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-seqs", type=int, default=100)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--signal-bit-boost", type=float, default=2.0)
    parser.add_argument("--qjl-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    calibrate(args.model, args.output, args.n_seqs, args.bits, args.signal_bit_boost, args.qjl_dim, args.device)
