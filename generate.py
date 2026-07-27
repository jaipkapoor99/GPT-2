"""
GPT-2 (124M) Text Generation CLI Script
Features:
Top-K Sampling and Temperature-controlled Autoregressive Generation with full CLI argument parser.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoTokenizer
from config import GPT2Config
from model import GPT2

def generate_text(model, tokenizer, config, start_str="Hello, I am a language model,", max_new_tokens=50, temperature=0.7, top_k=50, device="cuda"):
    model.eval()
    input_indices = tokenizer.encode(start_str)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_indices = input_indices[-config.T:]
            x = torch.tensor([context_indices], dtype=torch.long, device=device)
            
            logits, _ = model(x)
            next_token_logits = logits[:, -1, :] / temperature
            
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            
            idx = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(top_k_indices, -1, idx).item()
            input_indices.append(next_token)
            
    return tokenizer.decode(input_indices)

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Text Generation CLI (Accelerate Engine)")
    parser.add_argument("prompt", type=str, nargs="?", default="The future of artificial intelligence is", help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=60, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling cutoff")
    parser.add_argument("--load-checkpoint", type=str, default="accelerate_checkpoint", help="Path to Accelerate checkpoint directory")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    
    accelerator.print(f"=== ACCELERATE TEXT GENERATION ENGINE ===")
    accelerator.print(f"Loading SmolLM Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    config = GPT2Config(vocab_size=tokenizer.vocab_size)
    model = GPT2(config)
    model = accelerator.prepare(model)
    
    if os.path.isdir(args.load_checkpoint):
        accelerator.print(f"Loading Accelerate native state from '{args.load_checkpoint}'...")
        accelerator.load_state(args.load_checkpoint)
    elif os.path.exists("gpt2-fineweb-124m/model.safetensors"):
        from safetensors.torch import load_file
        accelerator.print(f"Loading Safetensors weights from 'gpt2-fineweb-124m/model.safetensors'...")
        state_dict = load_file("gpt2-fineweb-124m/model.safetensors")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(state_dict)
    else:
        accelerator.print(f"No checkpoint found! Running generation with initial model weights...")
        
    accelerator.print(f"\nPrompt: '{args.prompt}'")
    output = generate_text(model, tokenizer, config, start_str=args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, device=device)
    accelerator.print("\n--- GENERATED TEXT ---")
    accelerator.print(output)

if __name__ == "__main__":
    main()
