"""
Hugging Face Transformers Text Generation Script
Loads model weights directly from Hugging Face Hub (safetensors format) using the Hugging Face `transformers` library.
"""

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
    
    # Load model using Hugging Face transformers (automatically loads model.safetensors)
    model = AutoModelForCausalLM.from_pretrained(args.repo_id).to(device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    print(f"Model loaded successfully on [{device.upper()}].")
    print(f"\nPrompt: '{args.prompt}'")
    
    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    # Generate text using Hugging Face transformers pipeline
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- GENERATED TEXT (TRANSFORMERS + SAFETENSORS) ---")
    print(generated_text)

if __name__ == "__main__":
    main()
