"""
GPT-2 Configuration Module
Contains dataclass parameters for the 2026-standard GPT-2 (124M) architecture.
Dynamically populates vocab_size from dataset metadata (fineweb_meta.json or shards/fineweb_shard_0000_meta.json).
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GPT2Config:
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
    vocab_size: Optional[int] = None # Dynamically populated from metadata or tokenizer
    head_dim: int = field(init=False)

    def __post_init__(self):
        assert self.C % self.n_head == 0, f"Embedding dimension C ({self.C}) must be divisible by n_head ({self.n_head})"
        self.head_dim = self.C // self.n_head
        
        # Dynamically load vocab_size from dataset metadata if not explicitly specified
        if self.vocab_size is None:
            meta_candidates = [
                "fineweb_meta.json",
                os.path.join("shards", "fineweb_shard_0000_meta.json")
            ]
            for meta_file in meta_candidates:
                if os.path.exists(meta_file):
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                        self.vocab_size = meta.get("vocab_size", 49152)
                    break
                    
            if self.vocab_size is None:
                self.vocab_size = 49152 # Fallback default
