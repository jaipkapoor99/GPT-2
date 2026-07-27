"""
Supervised Fine-Tuning (SFT) / Instruction Tuning Script
Fine-tunes the base pre-trained SOTA GPT-2 model into an instruction-following chat assistant.
Supports large-scale instruction datasets:
  - HuggingFaceTB/smoltalk (~1.1 Million multi-turn ChatML conversations)
  - HuggingFaceH4/ultrachat_200k (208k multi-turn conversations)
  - tatsu-lab/alpaca (52k instruction pairs)
Uses ChatML formatting (<|im_start|> user / assistant <|im_end|>) with prompt token loss masking.
"""

import os
import math
import time
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
from accelerate import Accelerator

def parse_args():
    parser = argparse.ArgumentParser(description="Large-Scale Supervised Fine-Tuning (SFT) for GPT-2 SOTA")
    parser.add_argument("--model-path", type=str, default="gpt2-fineweb-124m", help="Path to base model directory or HF repo")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceTB/smoltalk", help="Instruction dataset name")
    parser.add_argument("--dataset-config", type=str, default="all", help="Dataset subset config (e.g., 'all' for smoltalk)")
    parser.add_argument("--max-samples", type=int, default=100000, help="Maximum training samples to load (set -1 for full dataset)")
    parser.add_argument("--output-dir", type=str, default="gpt2-sota-instruct", help="Directory to save fine-tuned instruct model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of SFT training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate for SFT")
    parser.add_argument("--max-len", type=int, default=512, help="Maximum sequence length")
    return parser.parse_args()

class InstructionDataset(Dataset):
    """
    Formated Instruction Dataset supporting both multi-turn 'messages' and 'instruction/output' schemas
    with Completion-Only Loss Masking (<|im_start|> tags).
    """
    def __init__(self, raw_data, tokenizer, max_len=512):
        self.samples = []
        
        for item in raw_data:
            # Handle multi-turn conversation format (e.g. smoltalk / ultrachat)
            if "messages" in item and isinstance(item["messages"], list):
                messages = item["messages"]
                input_ids = []
                labels = []
                
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "").strip()
                    if not content:
                        continue
                        
                    turn_str = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                    turn_ids = tokenizer.encode(turn_str, add_special_tokens=False)
                    
                    input_ids.extend(turn_ids)
                    if role == "assistant":
                        labels.extend(turn_ids)
                    else:
                        labels.extend([-100] * len(turn_ids))
                        
                if not input_ids or not any(l != -100 for l in labels):
                    continue
                    
                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    labels = labels[:max_len]
                    
                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                })
            else:
                # Handle single-turn instruction/input/output format (e.g. alpaca / openorca)
                instruction = item.get("instruction", "").strip()
                inp = item.get("input", "").strip()
                output = item.get("output", "").strip()
                
                if not instruction or not output:
                    continue
                    
                user_text = f"Instruction: {instruction}" + (f"\nInput: {inp}" if inp else "")
                prompt_str = f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
                response_str = f"{output}<|im_end|>"
                
                prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
                response_ids = tokenizer.encode(response_str, add_special_tokens=False)
                
                input_ids = prompt_ids + response_ids
                if len(input_ids) > max_len:
                    input_ids = input_ids[:max_len]
                    prompt_len = min(len(prompt_ids), max_len)
                else:
                    prompt_len = len(prompt_ids)
                    
                labels = [-100] * prompt_len + input_ids[prompt_len:]
                
                self.samples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long)
                })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def pad_collate_fn(batch, pad_token_id=0):
    max_len = max(len(s["input_ids"]) for s in batch)
    
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    
    for s in batch:
        ids = s["input_ids"]
        labs = s["labels"]
        pad_len = max_len - len(ids)
        
        padded_ids = torch.cat([ids, torch.tensor([pad_token_id] * pad_len, dtype=torch.long)])
        padded_labs = torch.cat([labs, torch.tensor([-100] * pad_len, dtype=torch.long)])
        attn_mask = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
        
        input_ids_list.append(padded_ids)
        labels_list.append(padded_labs)
        attention_mask_list.append(attn_mask)
        
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attention_mask_list)
    }

def main():
    args = parse_args()
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="bf16")
    
    accelerator.print(f"=== LARGE-SCALE SUPERVISED FINE-TUNING (SFT) ===")
    accelerator.print(f"Base Model: {args.model_path}")
    accelerator.print(f"Dataset: {args.dataset_name} (max_samples: {args.max_samples})")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Load instruction dataset
    try:
        if args.dataset_config and args.dataset_name == "HuggingFaceTB/smoltalk":
            raw_ds = load_dataset(args.dataset_name, args.dataset_config, split="train")
        else:
            raw_ds = load_dataset(args.dataset_name, split="train")
    except Exception as e:
        accelerator.print(f"Falling back to default split for dataset {args.dataset_name}: {e}")
        raw_ds = load_dataset(args.dataset_name, split="train")
        
    if args.max_samples > 0 and len(raw_ds) > args.max_samples:
        raw_ds = raw_ds.select(range(args.max_samples))
        
    sft_ds = InstructionDataset(raw_ds, tokenizer, max_len=args.max_len)
    
    train_loader = DataLoader(
        sft_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: pad_collate_fn(b, pad_token_id=tokenizer.pad_token_id)
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    num_training_steps = len(train_loader) * args.epochs // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03 * num_training_steps), num_training_steps=num_training_steps)
    
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    
    accelerator.print(f"Starting SFT Fine-Tuning ({len(sft_ds):,} processed samples, {args.epochs} epoch(s), {num_training_steps:,} optimizer steps)...")
    
    step = 0
    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                step += 1
                if step % 50 == 0 or step == num_training_steps:
                    accelerator.print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step:,}/{num_training_steps:,}] | SFT Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
                    
    accelerator.print("\n✓ Large-Scale SFT Fine-Tuning Complete!")
    
    # Save fine-tuned instruct model
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)
        accelerator.print(f"✓ Fine-tuned Instruct Model saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()
