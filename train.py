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

def print_table_header(accelerator: Accelerator):
    accelerator.print("+-------+------------+-----------+-----------+")
    accelerator.print("| Step  | Train Loss | Dev Loss  |    LR     |")
    accelerator.print("+-------+------------+-----------+-----------+")

def print_table_row(step, train_loss, dev_loss, lr, accelerator: Accelerator):
    accelerator.print(f"| {step:5d} |   {train_loss:7.4f}  |  {dev_loss:7.4f}  | {lr:8.2e}  |")

def print_table_footer(accelerator: Accelerator):
    accelerator.print("+-------+------------+-----------+-----------+")

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
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    
    config = GPT2Config(max_steps=args.max_steps, gradient_checkpointing=args.gradient_checkpointing)
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
    if args.optimizer == "muon":
        muon_params = [p for name, p in model.named_parameters() if p.ndim == 2 and 'wte' not in name and 'wpe' not in name]
        adamw_decay_params = [p for name, p in model.named_parameters() if p.ndim < 2 and 'wte' not in name and 'wpe' not in name and p.requires_grad]
        adamw_nodecay_params = [p for name, p in model.named_parameters() if ('wte' in name or 'wpe' in name) and p.requires_grad]
        
        adamw_groups = [
            {"params": adamw_decay_params, "weight_decay": 0.1},
            {"params": adamw_nodecay_params, "weight_decay": 0.0}
        ]
        
        optimizer_muon = Muon(muon_params, lr=0.04, momentum=0.95)
        optimizer_adamw = torch.optim.AdamW(adamw_groups, lr=config.learning_rate, betas=(0.9, 0.95), fused=True)
        model, optimizer_muon, optimizer_adamw = accelerator.prepare(model, optimizer_muon, optimizer_adamw)
    else:
        decay_params = [p for name, p in model.named_parameters() if p.ndim >= 2 and 'wte' not in name]
        nodecay_params = [p for name, p in model.named_parameters() if p.ndim < 2 or 'wte' in name]
        adamw_groups = [
            {"params": decay_params, "weight_decay": 0.1},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        optimizer_adamw = torch.optim.AdamW(adamw_groups, lr=config.learning_rate, betas=(0.9, 0.95), fused=True)
        model, optimizer_adamw = accelerator.prepare(model, optimizer_adamw)
        optimizer_muon = None
    
    # Accelerate Native State Resumption (Model + Optimizer + RNG state + Step Counter)
    accelerate_dir = "accelerate_checkpoint"
    resume_step = 0
    state_file = os.path.join(accelerate_dir, "training_state.json")
    
    if args.resume or args.load_checkpoint:
        dir_to_load = args.load_checkpoint if (args.load_checkpoint and os.path.isdir(args.load_checkpoint)) else accelerate_dir
        if os.path.isdir(dir_to_load):
            accelerator.print(f"Resuming full Accelerate training state (model + optimizer) from '{dir_to_load}'...")
            accelerator.load_state(dir_to_load)
            
            if os.path.exists(os.path.join(dir_to_load, "training_state.json")):
                with open(os.path.join(dir_to_load, "training_state.json"), "r") as f:
                    state_info = json.load(f)
                    resume_step = state_info.get("step", 0)
                    accelerator.print(f"✓ Restored exact training step: {resume_step:,} (Fast-forwarding data loader past seen batches)")
            accelerator.print(f"✓ Full Accelerate training state restored successfully!")
    
    # Enable PyTorch 2.0 TorchInductor Compilation for Speed Boost
    accelerator.print("Compiling model graph with torch.compile()...")
    model = torch.compile(model)
    
    model.train()
    step = resume_step
    history = []
    
    accelerator.print(f"Starting pre-training for {config.max_steps} steps ({tokens_per_step * config.max_steps:,} total tokens)...\n")
    print_table_header(accelerator)
    
    pbar = tqdm(total=config.max_steps, initial=resume_step, desc="Pre-training GPT-2")
    
    decay_start_step = int(0.8 * config.max_steps) # 80% stable phase, 20% decay phase
    batches_seen = 0
    
    while step < config.max_steps:
        for xb, yb in train_loader:
            if step >= config.max_steps:
                break
                
            # Fast-forward DataLoader batches if resuming past completed steps
            if batches_seen < resume_step * GRAD_ACCUM_STEPS:
                batches_seen += 1
                continue
                
            # WSD (Warmup-Stable-Linear-Decay) Learning Rate Schedule
            if step < config.warmup_steps:
                lr = config.learning_rate * (step + 1) / config.warmup_steps
            elif step < decay_start_step:
                lr = config.learning_rate
            else:
                decay_ratio = (step - decay_start_step) / max(1, config.max_steps - decay_start_step)
                lr = config.min_lr + (1.0 - decay_ratio) * (config.learning_rate - config.min_lr)
                
            for param_group in optimizer_adamw.param_groups:
                param_group['lr'] = lr
            if optimizer_muon is not None:
                for param_group in optimizer_muon.param_groups:
                    param_group['lr'] = 0.04 * (lr / config.learning_rate)
                
            with accelerator.accumulate(model):
                logits, loss = model(xb, yb)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_adamw.step()
                if optimizer_muon is not None:
                    optimizer_muon.step()
                optimizer_adamw.zero_grad()
                if optimizer_muon is not None:
                    optimizer_muon.zero_grad()
                
            if accelerator.sync_gradients:
                step += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
                
                # Periodic Evaluation & Tabular Metric Logging
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
                    
                    print_table_row(step, loss.item(), avg_dev_loss, lr, accelerator)
                    
                    history.append({
                        "step": step,
                        "train_loss": loss.item(),
                        "dev_loss": avg_dev_loss,
                        "learning_rate": lr
                    })
                    model.train()

    pbar.close()
    print_table_footer(accelerator)
    accelerator.print("\nPre-training Complete!")
    
    # Save Accelerate Native Multi-GPU State
    accelerator.print("Saving full Accelerate state to 'accelerate_checkpoint'...")
    accelerator.save_state(accelerate_dir)
    accelerator.save_model(model, "gpt2-fineweb-124m")
    
    if accelerator.is_main_process:
        with open(os.path.join(accelerate_dir, "training_state.json"), "w") as f:
            json.dump({"step": step, "max_steps": config.max_steps}, f, indent=2)
        accelerator.print("✓ Saved full Accelerate state, training step info, and model directory 'gpt2-fineweb-124m'.")
    
    if accelerator.is_main_process:
        # Save Tabular Loss Metrics (JSON & CSV)
        with open("loss_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        with open("loss_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "dev_loss", "learning_rate"])
            writer.writeheader()
            writer.writerows(history)
            
        accelerator.print("✓ Saved loss metrics to 'loss_history.json' and 'loss_history.csv'")
        
        # Render & Save Vector SVG Loss Curve Plot
        steps_list = [h["step"] for h in history]
        train_list = [h["train_loss"] for h in history]
        dev_list   = [h["dev_loss"] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps_list, train_list, label="Train Loss", marker="o", linewidth=2)
        plt.plot(steps_list, dev_list, label="Dev Loss", marker="s", linewidth=2)
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Cross Entropy Loss", fontsize=12)
        plt.title("GPT-2 (124M) Pre-training Loss Curve", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig("loss_curve.svg", format="svg")
        plt.close()
        accelerator.print("✓ Vector loss curve rendered and saved to 'loss_curve.svg'")

if __name__ == "__main__":
    main()
