"""
GPT-2 (124M) Text Generation CLI Script
Features:
Top-K Sampling and Temperature-controlled Autoregressive Generation with full CLI argument parser.
"""

import argparse
import os
import torch
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="GPT-2 Text Generation CLI")
    parser.add_argument("prompt", type=str, nargs="?", default="The future of artificial intelligence is", help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=60, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K sampling cutoff")
    parser.add_argument("--load-checkpoint", type=str, default="gpt2_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SmolLM Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    config = GPT2Config(vocab_size=tokenizer.vocab_size)
    model = GPT2(config).to(device)
    
    if os.path.exists(args.load_checkpoint):
        print(f"Loading trained weights from '{args.load_checkpoint}'...")
        state_dict = torch.load(args.load_checkpoint, map_location=device)
        cleaned_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_state_dict)
    else:
        print(f"Checkpoint '{args.load_checkpoint}' not found! Running generation with initial weights...")
        
    print(f"\nPrompt: '{args.prompt}'")
    output = generate_text(model, tokenizer, config, start_str=args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, device=device)
    print("\n--- GENERATED TEXT ---")
    print(output)

if __name__ == "__main__":
    main()
