"""
Ultron Configuration Module
Contains dataclass parameters for the 2026-standard Ultron (124M) architecture.
All hyper-parameter fields have sensible 2026 SOTA defaults.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class UltronConfig:
    B: int = 16                 # Micro-batch size per pass (65k tokens/step with grad accum 4)
    grad_accum_steps: int = 4   # Gradient accumulation steps
    T: int = 1024               # Time / Sequence length (T)
    C: int = 768                # Channels / Embedding dimension (C)
    n_head: int = 12            # Number of self-attention query heads
    n_kv_head: int = 4          # Number of key/value heads for GQA (LLaMA 3 / Qwen 2.5 style)
    n_layer: int = 12           # Number of stacked Transformer blocks
    dropout: float = 0.0        # No dropout (modern pre-training standard)
    learning_rate: float = 1.2e-3 # Learning rate
    min_lr: float = 1.2e-4        # Minimum learning rate
    warmup_steps: int = 200     # Warmup steps
    max_steps: int = 152587     # Total training steps (~10B tokens at 65k tokens/step)
    eval_interval: int = 250    # Evaluation interval

    rope_base: float = 10000.0  # RoPE frequency base parameter
    logit_softcap: float = 15.0 # Gemma 2 standard logit softcapping (prevents overconfidence)
    vocab_size: int = 49152      # SmolLM BPE Vocab size
    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.C % self.n_head == 0, f"Embedding dimension C ({self.C}) must be divisible by n_head ({self.n_head})"
        assert self.n_head % self.n_kv_head == 0, f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
        self.head_dim = self.C // self.n_head

    @classmethod
    def from_metadata(cls, **overrides):
        """Create an UltronConfig instance with optional overrides."""
        return cls(**overrides)

