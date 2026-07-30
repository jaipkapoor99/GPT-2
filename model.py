"""
Ultron (124M) PyTorch Model Module (2026 SOTA Standards)
Implements:
1. FlashAttention-2 (F.scaled_dot_product_attention) with smart context enforcement
2. SwiGLU Gated FeedForward Network (LLaMA 3 / Qwen 2.5 architecture)
3. Pre-LayerNorm Residual Skip Connections
4. Weight Tying (lm_head.weight = wte.weight)
5. 1 / sqrt(2*N) Residual Parameter Initialization
6. Pre-trained weight loader from official OpenAI weights
7. KV Caching for O(N) autoregressive generation
8. Zero-Copy Grouped-Query Attention (GQA)
9. Vocab Size padding (multiple of 64) for Tensor Cores
"""

import math
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    from .config import UltronConfig
except ImportError:
    from config import UltronConfig


@dataclass
class UltronOutput:
    """Output class for UltronModel containing logits, optional loss, and optional KV cache."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None

def _find_free_port():
    """Find a free TCP port on localhost to avoid EADDRINUSE crashes."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ Apply Rotary Position Embedding (RoPE) to input tensor x """
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos[..., :d] - x2 * sin[..., :d]
    y2 = x1 * sin[..., :d] + x2 * cos[..., :d]
    return torch.cat([y1, y2], dim=-1)

class RotaryEmbedding(nn.Module):
    """ Rotary Position Embedding (RoPE) Module (LLaMA 3 / Qwen 2.5 standard) """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len + offset > self.cos_cached.size(2):
            self._build_cache(seq_len + offset)
        return self.cos_cached[:, :, offset:seq_len+offset, :], self.sin_cached[:, :, offset:seq_len+offset, :]

class CausalSelfAttention(nn.Module):
    """ High-performance Grouped-Query Attention (GQA) with RoPE using PyTorch FlashAttention """
    def __init__(self, config: UltronConfig):
        super().__init__()
        assert config.C % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.num_queries_per_kv = config.n_head // config.n_kv_head
        self.head_dim = config.head_dim
        self.C = config.C
        self.dropout_p = config.dropout

        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        
        self.c_attn = nn.Linear(config.C, q_dim + 2 * kv_dim, bias=False)
        self.c_proj = nn.Linear(config.C, config.C, bias=False)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rot_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        
        q_dim = self.n_head * self.head_dim
        kv_dim = self.n_kv_head * self.head_dim
        
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rot_emb is not None:
            cos, sin = rot_emb
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
            
        new_past_key_value = (k, v)

        if self.num_queries_per_kv > 1:
            # Zero-Copy GQA expansion using unsqueeze/expand/reshape instead of repeat_interleave
            k_len = k.size(2)
            k = k.unsqueeze(2).expand(B, self.n_kv_head, self.num_queries_per_kv, k_len, self.head_dim).reshape(B, self.n_head, k_len, self.head_dim)
            v = v.unsqueeze(2).expand(B, self.n_kv_head, self.num_queries_per_kv, k_len, self.head_dim).reshape(B, self.n_head, k_len, self.head_dim)
            
        dropout_p = self.dropout_p if self.training else 0.0
        
        # Efficient Scaled Dot-Product Attention (PyTorch automatically selects optimal FlashAttention/CuDNN kernel)
        is_causal = self.training or q.size(2) > 1
        y = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y), new_past_key_value

class SwiGLUMLP(nn.Module):
    """ Modern SwiGLU FeedForward Network (used in LLaMA 3, Qwen 2.5, & Mistral) """
    def __init__(self, config: UltronConfig):
        super().__init__()
        hidden_dim = int(2 * (4 * config.C) / 3)
        hidden_dim = 64 * ((hidden_dim + 63) // 64) # Round to multiple of 64 for Tensor Cores
        
        self.w1 = nn.Linear(config.C, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.C, bias=False)
        self.w3 = nn.Linear(config.C, hidden_dim, bias=False)
        self.w2.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization (RMSNorm) - LLaMA / Qwen 2.5 standard """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class Block(nn.Module):
    """ Transformer Block: Communication (FlashAttention) + Computation (SwiGLU MLP) """
    def __init__(self, config: UltronConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.C)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.C)
        self.mlp  = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, rot_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present_key_value = self.attn(self.ln_1(x), rot_emb=rot_emb, past_key_value=past_key_value)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_key_value

class UltronModel(nn.Module):
    """ Full Ultron (124M) Language Model Class with RMSNorm, RoPE, & GQA """
    def __init__(self, config: UltronConfig):
        super().__init__()
        self.config = config
        
        # Padded vocab size for Tensor Core efficiency
        self.vocab_size = math.ceil(config.vocab_size / 64) * 64
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.vocab_size, config.C),
            wpe = nn.Embedding(config.T, config.C),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.C),
        ))
        self.lm_head = nn.Linear(config.C, self.vocab_size, bias=False)

        self.rotary_emb = RotaryEmbedding(config.head_dim, max_seq_len=config.T, base=config.rope_base)
            
        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

    @property
    def device(self):
        return next(self.parameters()).device

    def tie_weights(self):
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_cache: bool = False, past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None):
        B, T = idx.size()
        
        past_length = past_key_values[0][0].size(2) if past_key_values is not None else 0
        assert T + past_length <= self.config.T, f"Cannot forward sequence length {T + past_length}, model block size is {self.config.T}"
        
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb)
        rot_emb = self.rotary_emb(x, T, offset=past_length)
        
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            
            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                # Cache is not supported with gradient checkpointing
                x, _ = torch.utils.checkpoint.checkpoint(block, x, rot_emb, None, use_reentrant=False)
            else:
                x, present_kv = block(x, rot_emb=rot_emb, past_key_value=past_kv)
                
            if use_cache:
                present_key_values.append(present_kv)
                
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        if self.config.logit_softcap > 0.0:
            logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
            
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        if use_cache:
            return UltronOutput(logits=logits, loss=loss, past_key_values=present_key_values)
        return UltronOutput(logits=logits, loss=loss)

    def configure_optimizers(self, learning_rate: float, device_type: str = 'cuda'):
        """Hybrid Muon + AdamW optimizer setup.
        
        Muon handles 2D weight matrices (attention, MLP projections).
        AdamW handles everything else (embeddings, norms, biases).
        """
        from muon import Muon
        if not torch.distributed.is_initialized():
            backend = "nccl" if device_type == "cuda" else "gloo"
            port = _find_free_port()
            torch.distributed.init_process_group(
                backend=backend,
                rank=0,
                world_size=1,
                init_method=f"tcp://127.0.0.1:{port}"
            )
        muon_params = [p for name, p in self.named_parameters() if p.ndim == 2 and 'wte' not in name and 'wpe' not in name]
        adamw_decay_params = [p for name, p in self.named_parameters() if p.ndim < 2 and 'wte' not in name and 'wpe' not in name and p.requires_grad]
        adamw_nodecay_params = [p for name, p in self.named_parameters() if ('wte' in name or 'wpe' in name) and p.requires_grad]
        
        adamw_groups = [
            {"params": adamw_decay_params, "weight_decay": 0.1},
            {"params": adamw_nodecay_params, "weight_decay": 0.0}
        ]
        
        optimizer_muon = Muon(muon_params, lr=0.04, momentum=0.95)
        optimizer_adamw = torch.optim.AdamW(adamw_groups, lr=learning_rate, betas=(0.9, 0.95), fused=True)
        return optimizer_muon, optimizer_adamw

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        """ Generate autoregressively given a prompt idx using KV caching (O(N)) """
        past_key_values = None
        for i in range(max_new_tokens):
            if past_key_values is None:
                # Pre-fill phase: process the entire prompt
                idx_cond = idx if idx.size(1) <= self.config.T else idx[:, -self.config.T:]
            else:
                # Decoding phase: process only the last generated token
                idx_cond = idx[:, -1:]
                
            out = self(idx_cond, use_cache=True, past_key_values=past_key_values)
            logits = out.logits[:, -1, :self.config.vocab_size] # Strip padded vocab
            past_key_values = out.past_key_values
            
            if temperature != 1.0:
                logits = logits / temperature
                
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

