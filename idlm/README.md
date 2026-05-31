# I-DLM: Introspective Diffusion Language Model

Small-scale reproduction of [I-DLM](https://arxiv.org/abs/2604.11035) on WikiText-103.

**Method:** Fine-tune a frozen `rbf_ffn` AR checkpoint using the I-DLM
introspective-consistency objective, then decode with Introspective Strided
Decoding (ISD).

**Best AR base:** `rbf_ffn` SwiGLU + QK-norm + weight-norm → 58.16 val PPL

## Training

```bash
python -m idlm.train --config idlm/configs/baseline.yaml
```

## ISD Evaluation

```bash
python -m idlm.generate \
    --config idlm/configs/baseline.yaml \
    --checkpoint idlm/experiments/<run>/checkpoint_best.pt \
    --output results.jsonl
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| α | Introspective acceptance rate — how often the base AR model endorses ISD proposals |
| PPL | Perplexity of generated continuations vs WikiText-103 reference |
| TPF/OH | Tokens per forward / overhead — efficiency ratio; >1 means ISD beats sequential AR |

## Config Reference

| Key | Default | Description |
|-----|---------|-------------|
| `ar_checkpoint` | — | Path to `rbf_ffn` checkpoint `.pt` file |
| `lora_rank` | 8 | LoRA adapter rank |
| `lora_alpha` | 16.0 | LoRA scaling factor |
| `lora_target_modules` | `[q_proj, v_proj]` | Attention modules to apply LoRA |
| `seq_len` | 512 | Sequence length (model sees 2×seq_len during training) |
| `batch_size` | 8 | Training batch size |
| `max_steps` | 10000 | Training steps |
| `lr` | 3e-4 | AdamW learning rate (LoRA params only) |
| `stride` | 4 | ISD stride — tokens proposed per forward pass |
| `num_eval_examples` | 200 | Number of test sequences for ISD eval |
| `prompt_len` | 64 | Prompt prefix length for ISD generation |
| `gen_len` | 128 | Generation length for ISD evaluation |

## Reference

*Introspective Diffusion Language Models*
Yu, Jian, Wang, Zhou et al. — Together AI / UIUC / UT Austin / Princeton / Stanford
arXiv:2604.11035, Apr 2026
