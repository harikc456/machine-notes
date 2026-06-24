---
source_url: https://huggingface.co/collections/deepseek-ai/deepseek-v4
ingested: 2026-06-24
sha256: N/A (pdf-derived summary)
---

# DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence

**Authors:** DeepSeek-AI  
**Venue:** Technical report / arXiv preprint  

## Models

- **DeepSeek-V4-Pro:** 1.6T parameters, 49B activated, 1M token context
- **DeepSeek-V4-Flash:** 284B parameters, 13B activated, 1M token context

## Key Technical Contributions

### 1. Hybrid Attention: CSA + HCA
- **Compressed Sparse Attention (CSA):** Reduces attention complexity for long-context efficiency. Sparse attention pattern with compressed representations.
- **Heavily Compressed Attention (HCA):** Further compresses attention for long-context inference.
- Hybrid: most layers use CSA; selected layers use HCA for critical information retention.
- Result: V4-Pro requires only **27% of single-token inference FLOPs** vs DeepSeek-V3.2 at 1M context; KV cache 9.5–13.7× smaller

### 2. Manifold-Constrained Hyper-Connections (mHC)
- Replaces standard residual connections with doubly stochastic residual matrices (Sinkhorn-Knopp projected onto Birkhoff polytope)
- Restores identity mapping property; prevents training instability at scale
- See [[mhc]] entity page

### 3. Muon Optimizer
- Replaces AdamW for non-embedding parameters
- Based on Nesterov momentum + orthogonalization (Newton-Schulz iteration)
- Faster convergence and greater training stability vs AdamW

### 4. Training Scale
- Pre-trained on 32T+ diverse, high-quality tokens
- Comprehensive post-training pipeline (RL, instruction following, reasoning)
- DeepSeek-V4-Pro-Max (extended reasoning mode): redefines SOTA for open models

## Architecture (inherited from V3)
- DeepSeekMoE with auxiliary-loss-free load balancing
- Multi-head Latent Attention (MLA) from V3 (retained)

## Results

- Outperforms DeepSeek-V3.2 on core tasks
- V4-Pro-Max: competitive with Claude-Opus-4.6-Max, GPT-5.4-High, Gemini-3.1-Pro-High
- 1M token context routinely supported with manageable compute

## Relevance to Wiki

Primary source for [[deepseek-v4]] entity page. See [[mhc]] (manifold-constrained hyper-connections used here), [[mixture-of-experts]] (DeepSeekMoE foundation), [[kv-cache]] (CSA/HCA compress KV), [[deepseek-v3-2]] (predecessor).
