"""
GPT-2 (124M 2026 SOTA) Autoregressive Text Generation & Sampling Script
Features:
- Anti-Repetition Penalty (Frequency & Presence Penalty)
- Top-k (50) and Top-p (0.9) Nucleus Sampling
- Temperature Scaling
- Compatible with custom model.py and Hugging Face transformers AutoModelForCausalLM
"""

import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import GPT2Config
from model import GPT2

def sample_sequence(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cuda"
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window if needed
            cond_ids = generated[:, -1024:]
            
            # Forward pass
            out = model(cond_ids)
            logits = out.logits[:, -1, :]
                
            # Apply Repetition Penalty to prevent repetitive loops
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty
                        
            # Apply Temperature Scaling
            if temperature > 0:
                logits = logits / temperature
                
            # Apply Top-K Filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply Top-P (Nucleus) Filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
                
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat((generated, next_token), dim=1)
            
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="GPT-2 Text Generation with Anti-Repetition Safeguards")
    parser.add_argument("--prompt", type=str, default="Deep in the heart of the ancient forest", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering limit")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling cutoff")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty (1.15 disables repeating loops)")
    parser.add_argument("--use-hf", action="store_true", help="Load Hugging Face transformers model")
    parser.add_argument("--repo-id", type=str, default="jaipkapoor99/gpt2-2026-sota", help="Hugging Face model repository ID")
    parser.add_argument("--load-checkpoint", type=str, default="accelerate_checkpoint", help="Path to local Accelerate checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    if args.use_hf:
        model_path = args.repo_id
        print(f"Loading Hugging Face model from '{model_path}'...")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    else:
        config = GPT2Config(vocab_size=tokenizer.vocab_size)
        model = GPT2(config).to(device)
        if os.path.exists("gpt2-fineweb-124m/model.safetensors"):
            print("Loading custom weights from 'gpt2-fineweb-124m/model.safetensors'...")
            from safetensors.torch import load_file
            state_dict = load_file("gpt2-fineweb-124m/model.safetensors")
            clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(clean_state_dict, strict=False)
        elif os.path.isdir(args.load_checkpoint):
            print(f"Loading Accelerate checkpoint state from '{args.load_checkpoint}'...")
            from accelerate import Accelerator
            acc = Accelerator()
            model = acc.prepare(model)
            acc.load_state(args.load_checkpoint)
            
    print(f"\n--- PROMPT ---\n{args.prompt}\n")
    print("--- GENERATING (Anti-Repetition Enabled) ---")
    output_text = sample_sequence(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device
    )
    print(f"{output_text}\n")

if __name__ == "__main__":
    main()
