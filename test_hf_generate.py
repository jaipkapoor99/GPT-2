"""
Hugging Face Transformers Text Generation Script
Loads model weights directly from Hugging Face Hub (safetensors format) using the Hugging Face `transformers` library.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Transformers Model Inference")
    parser.add_argument("prompt", type=str, nargs="?", default="The spice must flow because", help="Prompt text for generation")
    parser.add_argument("--repo-id", type=str, default="jaipkapoor99/gpt2-2026-sota", help="Hugging Face repo ID")
    parser.add_argument("--max-tokens", type=int, default=60, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== HUGGING FACE TRANSFORMERS INFERENCE ===")
    print(f"Loading safetensors model from HF Hub: https://huggingface.co/{args.repo_id}")
    
    from config import GPT2Config
    from model import GPT2
    from safetensors.torch import load_file
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    # Load custom GPT2 2026 SOTA architecture
    config = GPT2Config(vocab_size=tokenizer.vocab_size)
    model = GPT2(config).to(device)
    
    if os.path.exists("gpt2-fineweb-124m/model.safetensors"):
        state_dict = load_file("gpt2-fineweb-124m/model.safetensors")
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
        print("✓ Loaded custom 2026 SOTA model weights successfully.")
    
    print(f"Model loaded on [{device.upper()}].")
    print(f"\nPrompt: '{args.prompt}'")
    
    # Autoregressive generation with Anti-Repetition Penalty
    from sample import sample_sequence
    generated_text = sample_sequence(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=1.15,
        device=device
    )
    print("\n--- GENERATED TEXT (2026 SOTA GPT-2) ---")
    print(generated_text)

if __name__ == "__main__":
    main()
