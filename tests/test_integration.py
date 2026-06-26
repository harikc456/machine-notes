# tests/test_integration.py
"""Slow end-to-end tests. Require GPU + model download.

Run with:
    pytest tests/test_integration.py --run-slow -v
"""
from __future__ import annotations
import pytest
import torch


@pytest.mark.slow
def test_turboquant_generation_and_ppl():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_quant import wrap, QuantConfig
    from kv_quant.bench.perplexity import compute_perplexity

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Baseline PPL
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    base_ppl = compute_perplexity(base, tokenizer, n_tokens=2048, chunk_size=256)
    del base
    torch.cuda.empty_cache()

    # TurboQuant @ 4 bits
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    cfg = QuantConfig(method="turboquant", bits=4)
    model = wrap(model, cfg)

    # Generation sanity
    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    assert len(generated.strip()) > 0, "Empty generation"

    # PPL within 1.5× baseline
    quant_ppl = compute_perplexity(model, tokenizer, n_tokens=2048, chunk_size=256)
    assert quant_ppl < base_ppl * 1.5, (
        f"TurboQuant PPL too high: {quant_ppl:.2f} vs baseline {base_ppl:.2f}"
    )


@pytest.mark.slow
def test_turboquant_memory_reduced():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kv_quant import wrap, QuantConfig
    from kv_quant.bench.memory import measure_kv_memory

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Baseline memory
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    base_mem = measure_kv_memory(base, tokenizer, max_new_tokens=100)
    del base
    torch.cuda.empty_cache()

    # TurboQuant @ 4 bits
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda"
    ).eval()
    model = wrap(model, QuantConfig(method="turboquant", bits=4))
    quant_mem = measure_kv_memory(model, tokenizer, max_new_tokens=100)

    # Compressed cache should be meaningfully smaller than fp16 estimate
    assert quant_mem["compression_ratio"] > 1.5, (
        f"Expected >1.5× compression, got {quant_mem['compression_ratio']:.2f}×"
    )
