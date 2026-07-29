"""
Spectacular Hugging Face Text Generation Script
Features:
- Beautiful terminal UI using `rich`
- Real-time token streaming using `TextStreamer`
- Best practices using `GenerationConfig`
"""

import argparse
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, GenerationConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.text import Text

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Spectacular GPT-2 Generation")
    parser.add_argument("--repo-id", type=str, default="jaipkapoor99/gpt2-2026-sota", help="Hugging Face repo ID")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling probability")
    parser.add_argument("--repetition-penalty", type=float, default=1.15, help="Repetition penalty")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(Panel.fit(
        f"[bold cyan]🤖 Spectacular Hugging Face Generator[/bold cyan]\n"
        f"Repo: [green]{args.repo_id}[/green]\n"
        f"Device: [yellow]{device.upper()}[/yellow]",
        border_style="cyan"
    ))

    with console.status("[bold cyan]Loading model and tokenizer online from Hugging Face Hub...", spinner="dots"):
        tokenizer = AutoTokenizer.from_pretrained(args.repo_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.repo_id, trust_remote_code=True).to(device)

    # Set pad_token_id properly if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create GenerationConfig using best practices
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_k=50,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Setup the text streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    console.print(f"[bold green]✓ Model loaded successfully![/bold green]\n")
    console.print("[dim]Type 'quit' or 'exit' to stop. Type 'config' to view generation settings.[/dim]\n")

    while True:
        try:
            prompt = Prompt.ask("[bold magenta]Prompt[/bold magenta]")
            if prompt.strip().lower() in ["quit", "exit"]:
                break
            if prompt.strip().lower() == "config":
                console.print(gen_config)
                continue
            if not prompt.strip():
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            console.print("\n[bold cyan]Output:[/bold cyan]", end=" ")
            
            # Streaming generation
            with torch.no_grad():
                model.generate(
                    **inputs,
                    generation_config=gen_config,
                    streamer=streamer
                )
            console.print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Generation interrupted. Type 'quit' to exit.[/yellow]\n")
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}\n")

if __name__ == "__main__":
    main()
