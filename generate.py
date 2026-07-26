"""
GPT-2 (124M) Text Generation CLI Script
Features:
Top-K Sampling and Temperature-controlled Autoregressive Generation
"""

import sys
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
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, I am a language model,"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading SmolLM Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    config = GPT2Config(vocab_size=tokenizer.vocab_size)
    model = GPT2(config).to(device)
    
    pth_checkpoint = "gpt2_model.pth"
    if os.path.exists(pth_checkpoint):
        print(f"Loading trained weights from '{pth_checkpoint}'...")
        model.load_state_dict(torch.load(pth_checkpoint, map_location=device))
    else:
        print("No trained checkpoint found! Running generation with initial weights...")
        
    print(f"\nPrompt: '{prompt}'")
    output = generate_text(model, tokenizer, config, start_str=prompt, max_new_tokens=60, temperature=0.7, top_k=50, device=device)
    print("\n--- GENERATED TEXT ---")
    print(output)

if __name__ == "__main__":
    main()
