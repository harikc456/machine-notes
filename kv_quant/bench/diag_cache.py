"""Quick diagnostic: check TurboQuantCache population after generate()."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_quant import wrap, QuantConfig

MODEL = "google/gemma-4-e2b-it"
PROMPT = "The quick brown fox jumps over the lazy dog."
MAX_NEW = 50  # short run

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
).eval()
device = next(model.parameters()).device

# --- Baseline ---
inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
torch.cuda.synchronize(device)
m0 = torch.cuda.memory_allocated(device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, return_dict_in_generate=True)
torch.cuda.synchronize(device)
m1 = torch.cuda.memory_allocated(device)
cache = out.past_key_values
seq = cache.get_seq_length() if cache is not None and hasattr(cache, "get_seq_length") else "N/A"
print(f"[baseline] cache={type(cache).__name__} seq_len={seq} delta={(m1-m0)/1e6:.2f}MB")

del out, model
torch.cuda.empty_cache()

# --- TurboQuant ---
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
).eval()
model = wrap(model, QuantConfig(method="turboquant", bits=4))

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
torch.cuda.synchronize(device)
m0 = torch.cuda.memory_allocated(device)
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, return_dict_in_generate=True)
torch.cuda.synchronize(device)
m1 = torch.cuda.memory_allocated(device)
cache = out.past_key_values
seq = cache.get_seq_length() if cache is not None and hasattr(cache, "get_seq_length") else "N/A"
print(f"[turboquant] cache={type(cache).__name__} seq_len={seq} delta={(m1-m0)/1e6:.2f}MB")

# If it's our cache, show buffer + compressed sizes
from kv_quant.turboquant import TurboQuantCache
if isinstance(cache, TurboQuantCache):
    buf_bytes = sum(
        t.nelement() * t.element_size()
        for buf in (cache._k_buf, cache._v_buf)
        for t in buf if t is not None
    )
    print(f"  buffer={buf_bytes/1e6:.2f}MB compressed={cache.compressed_bytes()/1e6:.2f}MB")
