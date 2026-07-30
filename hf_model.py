"""
hf_model.py — Hugging Face Integration Layer for Ultron (124M)

Defines UltronHFConfig and UltronForCausalLM to make Ultron compatible with
Hugging Face's AutoModelForCausalLM and PreTrainedModel interfaces.
"""

from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from model import UltronConfig, UltronModel, RotaryEmbedding


class UltronHFConfig(PretrainedConfig):
    """Hugging Face Config wrapper for Ultron architecture."""
    model_type = "ultron"

    def __init__(
        self,
        vocab_size: int = 49152,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_kv_head: int = 4,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        logit_softcap: float = 15.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.dropout = dropout
        self.rope_base = rope_base
        self.logit_softcap = logit_softcap
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.tie_word_embeddings = True

        super().__init__(**kwargs)

    def to_ultron_config(self) -> UltronConfig:
        """Convert HF config to internal UltronConfig dataclass."""
        return UltronConfig(
            vocab_size=self.vocab_size,
            T=self.n_positions,
            C=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_kv_head=self.n_kv_head,
            dropout=self.dropout,
            rope_base=self.rope_base,
            logit_softcap=self.logit_softcap,
        )


class UltronForCausalLM(PreTrainedModel, GenerationMixin):
    """Hugging Face PreTrainedModel wrapper around native UltronModel."""
    config_class = UltronHFConfig
    base_model_prefix = ""
    _supports_flash_attn_2 = True
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config: UltronHFConfig):
        super().__init__(config)
        ultron_cfg = config.to_ultron_config()
        self.transformer = UltronModel(ultron_cfg).transformer
        self.rotary_emb = RotaryEmbedding(ultron_cfg.head_dim, max_seq_len=ultron_cfg.T, base=ultron_cfg.rope_base)
        self.lm_head = nn.Linear(ultron_cfg.C, math.ceil(ultron_cfg.vocab_size / 64) * 64, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.transformer.wte = new_embeddings
        self.lm_head.weight = new_embeddings.weight

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module):
        self.lm_head = new_embeddings
        self.transformer.wte.weight = new_embeddings.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        B, T = input_ids.size()
        tok_emb = self.transformer.wte(input_ids)
        x = self.transformer.drop(tok_emb)
        past_len = past_key_values[0][0].size(2) if past_key_values is not None else 0
        rot_emb = self.rotary_emb(x, T, offset=past_len)

        new_past_kv = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, rot_emb=rot_emb, past_key_value=past_kv)
            if use_cache:
                new_past_kv.append(present_kv)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if getattr(self.config, "logit_softcap", 0.0) > 0.0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        output = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=tuple(new_past_kv) if use_cache else None,
        )

        if not return_dict:
            res = (output.logits,)
            if output.past_key_values is not None:
                res = res + (output.past_key_values,)
            if output.loss is not None:
                res = (output.loss,) + res
            return res

        return output

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, **kwargs
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }


# Register for AutoClass dynamic loading
UltronHFConfig.register_for_auto_class()
UltronForCausalLM.register_for_auto_class("AutoModelForCausalLM")
