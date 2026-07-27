"""
GPT-2 (124M 2026 SOTA) Benchmark & Zero-Shot Evaluation Suite
Measures:
1. Cross-Entropy Loss & Perplexity (PPL) on Dev Shards
2. Zero-Shot Multiple-Choice Completion (HellaSwag / Commonsense Reasoning)
3. Autoregressive KV-Cache Inference Generation Throughput (Tokens/Sec)
"""

import os
import math
import json
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator

from config import GPT2Config
from model import GPT2
from dataset import get_dataloaders

def forward_pass(model, input_ids, labels=None):
    """ Unified forward pass supporting both custom GPT2 model and Hugging Face transformers AutoModelForCausalLM """
    if hasattr(model, "config") and hasattr(model.config, "model_type") and getattr(model.config, "model_type") == "gpt2" and not hasattr(model.config, "head_dim"):
        out = model(input_ids=input_ids, labels=labels)
        return out.logits, out.loss
    else:
        return model(input_ids, targets=labels) if labels is not None else model(input_ids)

def evaluate_perplexity(model, dev_loader, accelerator, max_eval_batches=50):
    """ Evaluate average cross-entropy loss and calculate perplexity (PPL = exp(loss)) """
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for xb, yb in tqdm(dev_loader, desc="Evaluating Perplexity", disable=not accelerator.is_main_process):
            logits, loss = forward_pass(model, xb, labels=yb)
            total_loss += loss.item()
            total_batches += 1
            if total_batches >= max_eval_batches:
                break
                
    avg_loss = total_loss / max(1, total_batches)
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def evaluate_hellaswag_sample(model, tokenizer, device="cuda"):
    """
    Evaluates zero-shot multiple choice log-likelihood scoring on a sample HellaSwag context.
    """
    model.eval()
    context = "A person is playing basketball in a gym. They shoot the ball towards the hoop and"
    options = [
        " the ball swishes cleanly through the net for a three-point score.",
        " the car accelerates down the highway at high speed.",
        " the recipe requires three cups of flour and two eggs.",
        " the computer reboots into safe mode automatically."
    ]
    
    context_tokens = tokenizer.encode(context)
    scores = []
    
    with torch.no_grad():
        for opt in options:
            opt_tokens = tokenizer.encode(opt, add_special_tokens=False)
            full_tokens = context_tokens + opt_tokens
            x = torch.tensor([full_tokens[:-1]], dtype=torch.long, device=device)
            targets = torch.tensor([full_tokens[1:]], dtype=torch.long, device=device)
            
            logits, _ = forward_pass(model, x, labels=targets)
            
            # Compute log likelihood of the continuation tokens
            ctx_len = len(context_tokens) - 1
            opt_logits = logits[0, ctx_len:, :]
            opt_targets = targets[0, ctx_len:]
            
            log_probs = F.log_softmax(opt_logits, dim=-1)
            target_log_probs = log_probs[torch.arange(len(opt_targets)), opt_targets]
            
            score = target_log_probs.sum().item() / max(1, len(opt_tokens))
            scores.append(score)
            
    best_idx = int(torch.tensor(scores).argmax().item())
    return context, options, scores, best_idx

def benchmark_inference_speed(model, tokenizer, config, prompt="The future of artificial intelligence", max_tokens=100, device="cuda"):
    """ Measures autoregressive generation latency and throughput (Tokens / Second) """
    model.eval()
    input_indices = tokenizer.encode(prompt)
    x = torch.tensor([input_indices], dtype=torch.long, device=device)
    
    # Warmup pass
    with torch.no_grad():
        forward_pass(model, x)
        
    start_time = time.perf_counter()
    tokens_generated = 0
    
    with torch.no_grad():
        for _ in range(max_tokens):
            context_indices = input_indices[-config.T:]
            x_step = torch.tensor([context_indices], dtype=torch.long, device=device)
            logits, _ = forward_pass(model, x_step)
            next_token = logits[:, -1, :].argmax(dim=-1).item()
            input_indices.append(next_token)
            tokens_generated += 1
            
    end_time = time.perf_counter()
    total_time = end_time - start_time
    tokens_per_sec = tokens_generated / max(1e-5, total_time)
    
    return tokens_generated, total_time, tokens_per_sec, tokenizer.decode(input_indices)

def main():
    parser = argparse.ArgumentParser(description="GPT-2 SOTA Benchmark & Evaluation Suite (Transformers & Accelerate)")
    parser.add_argument("--repo-id", type=str, default="jaipkapoor99/gpt2-2026-sota", help="Hugging Face repo ID or local model path for transformers AutoModelForCausalLM")
    parser.add_argument("--load-checkpoint", type=str, default="accelerate_checkpoint", help="Path to Accelerate checkpoint or safetensors dir")
    parser.add_argument("--use-hf", action="store_true", help="Use Hugging Face transformers AutoModelForCausalLM for evaluation")
    parser.add_argument("--eval-batches", type=int, default=50, help="Number of dev batches for perplexity calculation")
    parser.add_argument("--bench-tokens", type=int, default=100, help="Number of tokens for throughput benchmark")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    
    accelerator.print("\n=======================================================")
    accelerator.print("   GPT-2 (124M 2026 SOTA) BENCHMARK & EVAL SUITE")
    accelerator.print("=======================================================")
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    if args.use_hf or os.path.exists(os.path.join("gpt2-fineweb-124m", "model.safetensors")):
        from transformers import AutoModelForCausalLM
        model_path = args.repo_id if args.use_hf else "gpt2-fineweb-124m"
        accelerator.print(f"Loading model via Hugging Face `transformers` from '{model_path}'...")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
            accelerator.print("✓ Loaded Hugging Face `transformers` model successfully.")
        except Exception as e:
            accelerator.print(f"Failed to load HF model: {e}. Falling back to Accelerate model...")
            config = GPT2Config(vocab_size=tokenizer.vocab_size)
            model = GPT2(config)
            model = accelerator.prepare(model)
            if os.path.isdir(args.load_checkpoint):
                accelerator.load_state(args.load_checkpoint)
    else:
        config = GPT2Config(vocab_size=tokenizer.vocab_size)
        model = GPT2(config)
        model = accelerator.prepare(model)
        if os.path.isdir(args.load_checkpoint):
            accelerator.print(f"Loading Accelerate checkpoint state from '{args.load_checkpoint}'...")
            accelerator.load_state(args.load_checkpoint)
            accelerator.print("✓ State loaded successfully.")
            
    # 1. Dev Perplexity Evaluation
    config = GPT2Config(vocab_size=tokenizer.vocab_size)
    _, dev_loader, _ = get_dataloaders(config, accelerator)
    dev_loss, dev_ppl = evaluate_perplexity(model, dev_loader, accelerator, max_eval_batches=args.eval_batches)
    
    accelerator.print("\n-------------------------------------------------------")
    accelerator.print(f"  Dev Cross-Entropy Loss : {dev_loss:.4f}")
    accelerator.print(f"  Dev Perplexity (PPL)   : {dev_ppl:.2f}")
    accelerator.print("-------------------------------------------------------")
    
    # 2. Zero-Shot HellaSwag Task Evaluation
    if accelerator.is_main_process:
        context, options, scores, best_idx = evaluate_hellaswag_sample(model, tokenizer, device=device)
        accelerator.print("\n--- ZERO-SHOT HELLASWAG REASONING TEST ---")
        accelerator.print(f"Context: '{context}'")
        for idx, (opt, score) in enumerate(zip(options, scores)):
            selected = "✓ [SELECTED]" if idx == best_idx else " "
            accelerator.print(f"  Option {idx+1} {selected}: {opt.strip()} (Normalized Log-Prob: {score:.4f})")
            
    # 3. Autoregressive Throughput Benchmark
    if accelerator.is_main_process:
        tokens_gen, total_time, tok_sec, text_out = benchmark_inference_speed(
            model, tokenizer, config, prompt="The desert planet of Arrakis is", max_tokens=args.bench_tokens, device=device
        )
        accelerator.print("\n--- INFERENCE THROUGHPUT BENCHMARK ---")
        accelerator.print(f"  Tokens Generated : {tokens_gen} tokens")
        accelerator.print(f"  Total Latency    : {total_time:.3f} seconds")
        accelerator.print(f"  Generation Speed : {tok_sec:.2f} tokens / second")
        accelerator.print("\nSample Output:")
        accelerator.print(f"'{text_out}'")
        
    # Save Benchmark Results JSON
    if accelerator.is_main_process:
        eval_results = {
            "dev_loss": dev_loss,
            "dev_perplexity": dev_ppl,
            "inference_tokens_per_sec": tok_sec,
            "hellaswag_sample_correct": (best_idx == 0)
        }
        with open("eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        accelerator.print("\n✓ Saved evaluation benchmark metrics to 'eval_results.json'.")
        accelerator.print("=======================================================\n")

if __name__ == "__main__":
    main()
