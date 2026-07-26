"""
GPT-2 (124M) Production Training CLI Script
Features:
1. Hugging Face Accelerate integration (FP16 mixed precision + Gradient Accumulation)
2. Cosine Learning Rate Schedule with Warmup
3. Periodic Validation Loss Evaluation (eval_interval)
4. Parameter Breakdown Reporting
"""

import math
import os
import torch
from accelerate import Accelerator
from tqdm import tqdm

from config import GPT2Config
from model import GPT2
from dataset import get_dataloaders

def print_parameter_breakdown(model: GPT2, accelerator: Accelerator):
    unwrapped = accelerator.unwrap_model(model)
    wte_params = sum(p.numel() for p in unwrapped.transformer.wte.parameters())
    wpe_params = sum(p.numel() for p in unwrapped.transformer.wpe.parameters())
    blocks_params = sum(p.numel() for p in unwrapped.transformer.h.parameters())
    ln_f_params = sum(p.numel() for p in unwrapped.transformer.ln_f.parameters())
    total_params = sum(p.numel() for p in unwrapped.parameters())
    
    accelerator.print("\n===========================================")
    accelerator.print("  GPT-2 (124M) DETAILED PARAMETER BREAKDOWN")
    accelerator.print("===========================================")
    accelerator.print(f"  Token Embeddings (wte)    : {wte_params:,} ({wte_params/total_params*100:.1f}%)")
    accelerator.print(f"  Position Embeddings (wpe) : {wpe_params:,} ({wpe_params/total_params*100:.1f}%)")
    accelerator.print(f"  12 Transformer Blocks (h) : {blocks_params:,} ({blocks_params/total_params*100:.1f}%)")
    accelerator.print(f"  Final LayerNorm (ln_f)    : {ln_f_params:,} (<0.1%)")
    accelerator.print(f"  Output LM Head (lm_head)  : 0 (Weight Tied with wte)")
    accelerator.print("-------------------------------------------")
    accelerator.print(f"  TOTAL TRAINABLE PARAMETERS: {total_params:,}")
    accelerator.print("===========================================\n")

def main():
    GRAD_ACCUM_STEPS = 8
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    
    config = GPT2Config()
    effective_batch = config.B * GRAD_ACCUM_STEPS
    tokens_per_step = effective_batch * config.T
    
    accelerator.print(f"Starting GPT-2 Pre-training Pipeline...")
    accelerator.print(f"  Micro-Batch Size (B): {config.B}")
    accelerator.print(f"  Grad Accum Steps    : {GRAD_ACCUM_STEPS}")
    accelerator.print(f"  Tokens / Step       : {tokens_per_step:,}")
    
    train_loader, dev_loader, _ = get_dataloaders(config, accelerator)
    
    model = GPT2(config)
    print_parameter_breakdown(model, accelerator)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    model, optimizer = accelerator.prepare(model, optimizer)
    
    model.train()
    step = 0
    pbar = tqdm(total=config.max_steps, desc="Training GPT-2 (124M)")
    
    while step < config.max_steps:
        for xb, yb in train_loader:
            if step >= config.max_steps:
                break
                
            # Cosine Learning Rate Schedule with Warmup
            if step < config.warmup_steps:
                lr = config.learning_rate * (step + 1) / config.warmup_steps
            else:
                decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                lr = config.min_lr + coeff * (config.learning_rate - config.min_lr)
                
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            with accelerator.accumulate(model):
                logits, loss = model(xb, yb)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
                
                if step % config.eval_interval == 0 or step == config.max_steps:
                    model.eval()
                    total_dev_loss = 0.0
                    dev_batches = 0
                    with torch.no_grad():
                        for xb_dev, yb_dev in dev_loader:
                            _, dev_loss = model(xb_dev, yb_dev)
                            total_dev_loss += dev_loss.item()
                            dev_batches += 1
                            if dev_batches >= 20:
                                break
                    avg_dev_loss = total_dev_loss / dev_batches
                    accelerator.print(f"Step {step:4d}/{config.max_steps} | Train Loss: {loss.item():.4f} | Dev Loss: {avg_dev_loss:.4f} | LR: {lr:.2e}")
                    model.train()

    pbar.close()
    accelerator.print("Training Complete!")
    
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), "gpt2_model.pth")
        accelerator.print("Model checkpoint saved to 'gpt2_model.pth'!")

if __name__ == "__main__":
    main()
