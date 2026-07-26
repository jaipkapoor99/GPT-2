"""
GPT-2 Configuration Module
Contains dataclass parameters for the 2026-standard GPT-2 (124M) architecture.
"""

from dataclasses import dataclass, field

@dataclass
class GPT2Config:
    num_documents: int = 50000  # FineWeb dataset documents to load
    B: int = 16                 # Micro-batch size per pass
    T: int = 1024               # Time / Sequence length (T)
    C: int = 768                # Channels / Embedding dimension (C)
    n_head: int = 12            # Number of self-attention heads
    n_layer: int = 12           # Number of stacked Transformer blocks
    dropout: float = 0.1        # 10% Dropout regularization
    learning_rate: float = 6e-4 # Max learning rate
    min_lr: float = 6e-5        # Min learning rate for cosine schedule
    warmup_steps: int = 200     # Warmup steps
    max_steps: int = 3000       # Total training steps
    eval_interval: int = 250    # Evaluate dev loss every 250 steps
    vocab_size: int = 49152     # Vocabulary size (SmolLM Byte-BPE)
    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.C % self.n_head == 0, f"Embedding dimension C ({self.C}) must be divisible by n_head ({self.n_head})"
        self.head_dim = self.C // self.n_head
