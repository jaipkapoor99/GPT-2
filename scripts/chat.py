"""
Interactive Terminal Chat Interface for GPT-2 SOTA Instruct Model
Allows real-time multi-turn conversation with ChatML formatting.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Interactive Chat CLI for GPT-2 SOTA Instruct")
    parser.add_argument("--model-path", type=str, default="gpt2-sota-instruct", help="Path to fine-tuned instruct model directory or HF repo")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== GPT-2 2026 SOTA INSTRUCT CHAT INTERFACE ===")
    print(f"Loading Instruct Model from: {args.model_path} on [{device.upper()}]...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    model.eval()

    print("✓ Model loaded successfully!")
    print("Type your message and press Enter. Type 'exit' or 'quit' to stop.\n")

    conversation_history = ""

    while True:
        try:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Format ChatML prompt
            prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            full_prompt = conversation_history + prompt

            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    repetition_penalty=1.15,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Extract generated response tokens
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Clean up any trailing chat tags
            response = response.replace("<|im_end|>", "").strip()

            print(f"\nAssistant > {response}\n")

            # Append to history for multi-turn chat
            conversation_history += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
