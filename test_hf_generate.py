"""
Hugging Face Transformers Remote Model Inference Script
Fetches the entire model (code, config, and safetensors weights) directly from Hugging Face Hub online.
Anybody on any machine can run this script to test the model!
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Remote Model Inference")
    parser.add_argument("prompt", type=str, nargs="?", default="The future of artificial intelligence is", help="Prompt text for generation")
    parser.add_argument("--repo-id", type=str, default="jaipkapoor99/gpt2-2026-sota", help="Hugging Face repo ID")
    parser.add_argument("--max-tokens", type=int, default=60, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== HUGGING FACE REMOTE MODEL INFERENCE ===")
    print(f"Fetching full model & tokenizer online from Hugging Face Hub: https://huggingface.co/{args.repo_id}")
    
    # Fetch full model architecture & weights online from Hugging Face Model Hub
    model = AutoModelForCausalLM.from_pretrained(args.repo_id, trust_remote_code=True, force_download=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    print(f"✓ Remote model fetched and loaded on [{device.upper()}].")
    print(f"\nPrompt: '{args.prompt}'")
    
    # Tokenize input
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    
    # Generate text using standard Hugging Face generate pipeline
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n--- GENERATED TEXT (FETCHED ONLINE FROM HUGGING FACE) ---")
    print(generated_text)

if __name__ == "__main__":
    main()
