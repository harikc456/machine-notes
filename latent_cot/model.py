from __future__ import annotations
import re
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


class ReasoningEncoder(nn.Module):
    """Compress a variable-length sequence of hidden states into K x d_z slots
    via K learnable queries that cross-attend the sequence. Runs in float32."""

    def __init__(self, d_model: int, n_slots: int, d_z: int, n_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        self.down = nn.Linear(d_model, d_z)

    def forward(self, hidden: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        B = hidden.size(0)
        q = self.ln_q(self.queries).unsqueeze(0).expand(B, -1, -1)
        kv = self.ln_kv(hidden)
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.down(attn_out)


class LatentCoTModel(nn.Module):
    """Shared LoRA backbone. For z conditions, prepends K soft-prefix embeddings
    (the projected reasoning encoding) to the question before the answer."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.condition = cfg.condition
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            cfg.backbone, torch_dtype=torch.bfloat16
        )
        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()  # needed for grad-checkpoint + peft
        target_modules = self._resolve_lora_targets(base, cfg.lora_targets)
        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=target_modules, task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(base, lora)
        # Multimodal checkpoints (e.g. gemma-4-E2B) nest the text decoder's
        # config under `text_config`; the top-level composite config has no
        # `hidden_size` of its own.
        text_config = getattr(base.config, "text_config", None)
        self.d_model = (
            text_config.hidden_size if text_config is not None else base.config.hidden_size
        )

        if self.condition in ("z", "z_shuffled"):
            self.encoder = ReasoningEncoder(
                self.d_model, cfg.n_slots, cfg.d_z, cfg.encoder_heads
            ).float()
            self.up = nn.Linear(cfg.d_z, self.d_model).float()

        self.to(self.device)

    @staticmethod
    def _resolve_lora_targets(base: nn.Module, lora_targets: list[str]) -> list[str] | str:
        """Scope LoRA target modules to the text decoder when the backbone is
        multimodal. Some checkpoints (e.g. gemma-4-E2B) reuse attention
        projection names (q_proj/k_proj/...) inside a vision tower whose
        linear layers are wrapped in a custom class peft's LoRA dispatcher
        cannot patch (`Gemma4ClippableLinear`). Matching those names by bare
        suffix would make peft try (and fail) to wrap the vision tower too,
        so when a `vision_tower` submodule is present we anchor the regex to
        `language_model` to target only the causal-LM attention layers."""
        has_vision_tower = any(
            "vision_tower" in name for name, _ in base.named_modules()
        )
        if not has_vision_tower:
            return lora_targets
        names = "|".join(re.escape(t) for t in lora_targets)
        return rf".*language_model.*\.({names})$"

    # ---- helpers -------------------------------------------------------
    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings()(ids)

    def _move(self, batch: dict) -> dict:
        return {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }

    def _encode_z(self, trace_ids, trace_mask) -> torch.Tensor:
        out = self.backbone(
            input_ids=trace_ids, attention_mask=trace_mask, output_hidden_states=True
        )
        hidden = out.hidden_states[-1]                 # (B, T, d_model) bf16
        kpm = trace_mask == 0                          # True = pad
        z = self.encoder(hidden, kpm)                  # fp32 (B, K, d_z)
        z_up = self.up(z)                              # fp32 (B, K, d_model)
        return z_up.to(self._embed(trace_ids[:, :1]).dtype)  # match embed dtype

    def trainable_parameters(self):
        params = [p for p in self.parameters() if p.requires_grad]
        return params

    def _placeholder_ids(self, batch_size: int, n: int) -> torch.Tensor:
        """Placeholder token ids for the z soft-prefix slots (not real
        vocabulary, any valid id works — pad_token_id is convenient)."""
        return torch.full(
            (batch_size, n), self.tokenizer.pad_token_id,
            dtype=torch.long, device=self.device,
        )

    def _per_layer_inputs(self, ids: torch.Tensor) -> torch.Tensor | None:
        """Precompute Gemma4's Per-Layer Embeddings (PLE) input from a REAL
        `input_ids` tensor. Passing this alongside `inputs_embeds` to the
        backbone avoids `Gemma4TextModel.get_per_layer_inputs`'s expensive
        (and CUDA-OOM-inducing) reverse embedding lookup, which only
        triggers when `input_ids` is None. Backbones without PLE support
        (no `language_model.get_per_layer_inputs`) return None, a no-op."""
        base = self.backbone.get_base_model()
        text_model = getattr(getattr(base, "model", base), "language_model", None)
        if text_model is None or not hasattr(text_model, "get_per_layer_inputs"):
            return None
        return text_model.get_per_layer_inputs(ids, None)

    # ---- forward (training) -------------------------------------------
    def forward(self, batch: dict) -> torch.Tensor:
        batch = self._move(batch)
        if self.condition in ("floor", "ceiling"):
            return self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss

        z_up = self._encode_z(batch["trace_ids"], batch["trace_mask"])
        q_emb = self._embed(batch["question_ids"])
        a_emb = self._embed(batch["answer_ids"])
        inputs_embeds = torch.cat([z_up, q_emb, a_emb], dim=1)

        B, K = z_up.shape[0], z_up.shape[1]
        z_mask = torch.ones(B, K, dtype=torch.long, device=self.device)
        attn = torch.cat([z_mask, batch["question_mask"], batch["answer_mask"]], dim=1)

        prefix_len = K + q_emb.size(1)
        ignore = torch.full((B, prefix_len), -100, dtype=torch.long, device=self.device)
        ans_labels = batch["answer_ids"].masked_fill(batch["answer_mask"] == 0, -100)
        labels = torch.cat([ignore, ans_labels], dim=1)

        z_ids = self._placeholder_ids(B, K)
        full_ids = torch.cat([z_ids, batch["question_ids"], batch["answer_ids"]], dim=1)
        per_layer_inputs = self._per_layer_inputs(full_ids)

        return self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels,
            per_layer_inputs=per_layer_inputs,
        ).loss

    # ---- generation (eval) --------------------------------------------
    @torch.no_grad()
    def generate(self, batch: dict, max_new_tokens: int) -> list[str]:
        batch = self._move(batch)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if self.condition in ("floor", "ceiling"):
            out = self.backbone.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"], **gen_kwargs,
            )
            gen = out[:, batch["input_ids"].size(1):]  # strip prompt
        else:
            z_up = self._encode_z(batch["trace_ids"], batch["trace_mask"])
            q_emb = self._embed(batch["question_ids"])
            inputs_embeds = torch.cat([z_up, q_emb], dim=1)
            B, K = z_up.shape[0], z_up.shape[1]
            z_mask = torch.ones(B, K, dtype=torch.long, device=self.device)
            attn = torch.cat([z_mask, batch["question_mask"]], dim=1)

            z_ids = self._placeholder_ids(B, K)
            full_ids = torch.cat([z_ids, batch["question_ids"]], dim=1)
            per_layer_inputs = self._per_layer_inputs(full_ids)

            # with inputs_embeds, generate() returns ONLY the new tokens
            gen = self.backbone.generate(
                inputs_embeds=inputs_embeds, attention_mask=attn,
                per_layer_inputs=per_layer_inputs, **gen_kwargs,
            )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
