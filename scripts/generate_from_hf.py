"""
generate_from_hf.py — Text generation pulling directly from Hugging Face Hub.

Pretends no local model or code exists by downloading UltronForCausalLM via trust_remote_code=True.

Usage:
    accelerate launch scripts/generate_from_hf.py [--prompt "TEXT"] [--max-new-tokens 50]
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from rich.console import Console

console = Console()

def main():
    if not any(k in os.environ for k in [
        "ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"
    ]):
        raise RuntimeError("Run with: accelerate launch scripts/generate_from_hf.py ...")

    parser = argparse.ArgumentParser(description="Generate text using Ultron from Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="jaipkapoor99/ultron-124m", help="Hugging Face Hub repository ID")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Prompt text for generation")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process:
        console.print(f"[bold cyan]🤗 Pulling model from Hugging Face Hub:[/bold cyan] [bold yellow]{args.repo_id}[/bold yellow]")

    # Load model from Hub using custom code registered in repo
    model = AutoModelForCausalLM.from_pretrained(
        args.repo_id,
        trust_remote_code=True,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    ).to(device)
    model.eval()

    # Load matching SmolLM BPE tokenizer (vocab size 49,152)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token

    if accelerator.is_main_process:
        console.print(f"[bold green]✨ Model loaded successfully on device:[/bold green] [bold yellow]{device}[/bold yellow]")
        console.print(f"[bold blue]📝 Input Prompt:[/bold blue] [italic]'{args.prompt}'[/italic]\n")

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if accelerator.is_main_process:
        console.print("[bold magenta]🤖 Generated Output:[/bold magenta]")
        console.print(f"[bold white]{generated_text}[/bold white]\n")


if __name__ == "__main__":
    main()
