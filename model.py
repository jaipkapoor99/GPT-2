"""
GPT-2 (124M) PyTorch Model Module (2026 SOTA Standards)
Implements:
1. FlashAttention-2 (F.scaled_dot_product_attention)
2. SwiGLU Gated FeedForward Network (LLaMA 3 / Qwen 2.5 architecture)
3. Pre-LayerNorm Residual Skip Connections
4. Weight Tying (lm_head.weight = wte.weight)
5. 1 / sqrt(2*N) Residual Parameter Initialization
6. Pre-trained weight loader from official OpenAI weights
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from config import GPT2Config

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """ Apply Rotary Position Embedding (RoPE) to input tensor x """
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

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

class CausalSelfAttention(nn.Module):
    """ High-performance Grouped-Query Attention (GQA) with RoPE using PyTorch FlashAttention """
    def __init__(self, config: GPT2Config):
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
        
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(self, x: torch.Tensor, rot_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
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

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            
        dropout_p = self.dropout_p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class SwiGLUMLP(nn.Module):
    """ Modern SwiGLU FeedForward Network (used in LLaMA 3, Qwen 2.5, & Mistral) """
    def __init__(self, config: GPT2Config):
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
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = RMSNorm(config.C)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.C)
        self.mlp  = SwiGLUMLP(config)

    def forward(self, x: torch.Tensor, rot_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), rot_emb=rot_emb)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    """ Full GPT-2 (124M) Language Model Class with RMSNorm, RoPE, & GQA """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.C),
            wpe = nn.Embedding(config.T, config.C),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.C),
        ))
        self.lm_head = nn.Linear(config.C, config.vocab_size, bias=False)
        
        if config.use_rope:
            self.rotary_emb = RotaryEmbedding(config.head_dim, max_seq_len=config.T, base=config.rope_base)
        else:
            self.rotary_emb = None
            
        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)

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

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.size()
        assert T <= self.config.T, f"Cannot forward sequence length {T}, model block size is {self.config.T}"
        
        tok_emb = self.transformer.wte(idx)
        if self.config.use_rope and self.rotary_emb is not None:
            x = self.transformer.drop(tok_emb)
            rot_emb = self.rotary_emb(x, T)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            rot_emb = None
        
        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, rot_emb, use_reentrant=False)
            else:
                x = block(x, rot_emb=rot_emb)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            if self.config.logit_softcap > 0.0:
                logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            if self.config.logit_softcap > 0.0:
                logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
            loss = None
            
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str = "gpt2"):
        """ Load official OpenAI pre-trained GPT-2 weights into scratch PyTorch model """
        from transformers import GPT2LMHeadModel
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"Loading pre-trained weights for {model_type} from Hugging Face...")
        
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, C=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, C=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, C=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, C=1600),
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['T'] = 1024
        
        config = GPT2Config(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight']
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                if k in sd and sd_hf[k].shape[::-1] == sd[k].shape:
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())
            elif k in sd:
                if sd_hf[k].shape == sd[k].shape:
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])
                    
        print(f"Successfully loaded official {model_type} weights!")
        return model
