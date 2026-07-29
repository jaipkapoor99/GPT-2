"""
GPT-2 (124M) Production Pre-training CLI Script (2026 SOTA Standards)
Features:
1. CLI arguments for --from-pretrained (load OpenAI weights or local checkpoint)
2. Tabulated loss metrics formatting
3. Loss tracking saved to CSV & JSON (loss_history.csv, loss_history.json)
4. Vector Loss Curve rendered & saved to loss_curve.svg
5. Fast, streamable PyTorch weight checkpoint saving (gpt2_model.pth)
"""

import argparse
import csv
import json
import math
import os
import matplotlib.pyplot as plt
import torch
from accelerate import Accelerator
from tqdm import tqdm

from config import GPT2Config
from model import GPT2
from dataset import get_dataloaders
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muon import Muon


def print_parameter_breakdown(model: GPT2, accelerator: Accelerator):
    unwrapped = accelerator.unwrap_model(model)
    wte_params = sum(p.numel() for p in unwrapped.transformer.wte.parameters())
    wpe_params = sum(p.numel() for p in unwrapped.transformer.wpe.parameters()) if not unwrapped.config.use_rope else 0
    blocks_params = sum(p.numel() for p in unwrapped.transformer.h.parameters())
    ln_f_params = sum(p.numel() for p in unwrapped.transformer.ln_f.parameters())
    total_params = sum(p.numel() for p in unwrapped.parameters())
    
    accelerator.print("\n===========================================")
    accelerator.print("  GPT-2 (124M 2026 SOTA) PARAMETER BREAKDOWN")
    accelerator.print("===========================================")
    accelerator.print(f"  Token Embeddings (wte)    : {wte_params:,} ({wte_params/total_params*100:.1f}%)")
    if unwrapped.config.use_rope:
        accelerator.print(f"  Positional Embeddings     : RoPE (Rotary Position Embeddings - 0 static params)")
    else:
        accelerator.print(f"  Position Embeddings (wpe) : {wpe_params:,} ({wpe_params/total_params*100:.1f}%)")
    accelerator.print(f"  12 Transformer Blocks (h) : {blocks_params:,} ({blocks_params/total_params*100:.1f}%) [GQA + SwiGLU + RMSNorm]")
    accelerator.print(f"  Final RMSNorm (ln_f)      : {ln_f_params:,} (<0.1%)")
    accelerator.print(f"  Output LM Head (lm_head)  : 0 (Weight Tied with wte)")
    accelerator.print("-------------------------------------------")
    accelerator.print(f"  TOTAL TRAINABLE PARAMETERS: {total_params:,}")
    accelerator.print("===========================================\n")

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Pre-training Pipeline")
    parser.add_argument("--from-pretrained", type=str, default=None, choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help="Optionally load pre-trained OpenAI weights instead of training from scratch")
    parser.add_argument("--resume", action="store_true", help="Resume training using Accelerate state or local 'gpt2_model.pth'")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Path to local checkpoint file or Accelerate checkpoint directory")
    parser.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw"], help="Optimizer choice: muon (Muon + AdamW hybrid) or adamw")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable activation checkpointing for ~60%% VRAM memory savings")
    parser.add_argument("--max-steps", type=int, default=3000, help="Total training steps")
    args = parser.parse_args()

    GRAD_ACCUM_STEPS = 4  # Micro-batch 16 * 4 accum passes = 65,536 tokens/step
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=GRAD_ACCUM_STEPS, log_with="wandb")
    
    config = GPT2Config(max_steps=args.max_steps, gradient_checkpointing=args.gradient_checkpointing)
    import dataclasses
    accelerator.init_trackers("gpt2-pretraining", config=dataclasses.asdict(config))
    effective_batch = config.B * GRAD_ACCUM_STEPS
    tokens_per_step = effective_batch * config.T
    
    accelerator.print(f"=== GPT-2 PRE-TRAINING PIPELINE ===")
    accelerator.print(f"  Micro-Batch Size (B): {config.B}")
    accelerator.print(f"  Grad Accum Steps    : {GRAD_ACCUM_STEPS}")
    accelerator.print(f"  Tokens / Step       : {tokens_per_step:,}")
    accelerator.print(f"  Total Steps         : {config.max_steps:,}")
    accelerator.print(f"  LR Schedule         : WSD (Warmup-Stable-Linear-Decay)")
    accelerator.print(f"  Optimizer           : {args.optimizer.upper()}")
    accelerator.print(f"  Positional Embed    : RoPE (Rotary Position Embeddings)")
    accelerator.print(f"  QK Head Normalizing : QK-Norm (RMSNorm on Q and K heads)")
    accelerator.print(f"  Logit Softcapping   : {config.logit_softcap} (Gemma 2 Standard)")
    accelerator.print(f"  Grad Checkpointing  : {'ENABLED (~60% VRAM savings)' if config.gradient_checkpointing else 'DISABLED'}")
    
    train_loader, dev_loader, _ = get_dataloaders(config, accelerator)
    
    # Model Initialization
    if args.from_pretrained:
        accelerator.print(f"\nLoading pre-trained OpenAI weights: '{args.from_pretrained}'...")
        model = GPT2.from_pretrained(args.from_pretrained)
    else:
        model = GPT2(config)
            
    print_parameter_breakdown(model, accelerator)
    
    torch.set_float32_matmul_precision('high')
    
    # Optimizer Construction (Accelerated Muon + Selective Zero-WD AdamW)
    optimizer_muon, optimizer_adamw = model.configure_optimizers(args.optimizer, config.learning_rate)
    
    if optimizer_muon is not None:
        model, optimizer_muon, optimizer_adamw = accelerator.prepare(model, optimizer_muon, optimizer_adamw)
    else:
        model, optimizer_adamw = accelerator.prepare(model, optimizer_adamw)
        
    # Enable PyTorch 2.0 TorchInductor Compilation for Speed Boost
    accelerator.print("Compiling model graph with torch.compile()...")
    model = torch.compile(model)
    
    from trainer import GPT2Trainer
    trainer = GPT2Trainer(model, optimizer_muon, optimizer_adamw, train_loader, dev_loader, config, accelerator)
    
    trainer.load_checkpoint(args.resume, args.load_checkpoint)
    trainer.train()
    accelerator.end_training()

if __name__ == "__main__":
    main()
