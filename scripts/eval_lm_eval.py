"""
scripts/eval_lm_eval.py — Run EleutherAI lm-evaluation-harness benchmarks on local Ultron checkpoint.

Downloads standard benchmark tasks online (Hellaswag, LAMBADA, PIQA, ARC, etc.) 
and evaluates the local Ultron model weights in-memory.

Usage:
    accelerate launch scripts/eval_lm_eval.py --checkpoint=accelerate_checkpoint --tasks=hellaswag,lambada_openai,piqa,arc_easy,arc_challenge --limit=100
"""

import argparse
import os
import sys
import torch
from accelerate import Accelerator
from rich.console import Console
from rich.table import Table

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import UltronModel
from hf_model import UltronHFConfig, UltronForCausalLM
from generate import load_model_weights
from transformers import AutoTokenizer

console = Console()
accelerator = Accelerator()


def main():
    if not any(k in os.environ for k in [
        "ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"
    ]):
        raise RuntimeError("Run with: accelerate launch scripts/eval_lm_eval.py ...")

    TASK_SUITES = {
        "reasoning": ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu", "drop"],
        "math": ["gsm8k", "minerva_math"],
        "coding": ["humaneval", "mbpp"],
        "knowledge": ["lambada_openai", "truthfulqa_mc1", "triviaqa"],
        "custom": ["custom_test_set"],
        "test_set": ["custom_test_set"],
    }
    TASK_SUITES["all"] = [task for name, suite in TASK_SUITES.items() if name not in ("custom", "test_set") for task in suite]

    parser = argparse.ArgumentParser(description="Evaluate local Ultron model using lm-evaluation-harness")
    parser.add_argument("--checkpoint", type=str, default="accelerate_checkpoint", help="Path to local checkpoint directory")
    parser.add_argument("--suite", type=str, choices=list(TASK_SUITES.keys()), default=None, help="Preset benchmark suite (reasoning, math, coding, knowledge, custom, test_set, all)")
    parser.add_argument("--tasks", type=str, default=None, help="Comma-separated list of lm-eval tasks (overrides --suite)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples/batches per task (for fast debugging)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU for evaluation")
    args = parser.parse_args()

    if args.tasks:
        task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    elif args.suite:
        task_list = TASK_SUITES[args.suite]
    else:
        task_list = ["hellaswag", "lambada_openai", "piqa", "arc_easy", "arc_challenge"]

    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.evaluator import simple_evaluate
    except ImportError:
        raise ImportError(
            "lm-eval is required for running benchmarks. Install via: pip install lm_eval"
        )

    # 1. Load local model weights
    if accelerator.is_main_process:
        console.print(f"[bold cyan]⚙️ Loading local model weights from '{args.checkpoint}'...[/bold cyan]")

    native_config = UltronConfig()
    native_model = UltronModel(native_config)
    native_model = load_model_weights(native_model, args.checkpoint)

    if task_list == ["custom_test_set"]:
        if accelerator.is_main_process:
            console.print("[bold blue]📊 Evaluating Ultron model on local test dataset split...[/bold blue]")

        from dataset import get_dataloaders
        import math

        _, _, test_loader = get_dataloaders(native_config, accelerator)
        total_loss = 0.0
        total_batches = 0

        native_model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if args.limit and i >= args.limit:
                    break
                out = native_model(x, targets=y)
                total_loss += out.loss.item()
                total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(avg_loss)
        bpc = avg_loss / math.log(2)

        if accelerator.is_main_process:
            table = Table(title="🏆 Ultron Custom Test Set Evaluation", show_header=True, header_style="bold magenta")
            table.add_column("Dataset Split", style="cyan")
            table.add_column("Metric", style="yellow")
            table.add_column("Score", style="bold green")

            table.add_row("local_test_set", "cross_entropy_loss", f"{avg_loss:.4f}")
            table.add_row("local_test_set", "perplexity", f"{perplexity:.4f}")
            table.add_row("local_test_set", "bits_per_character", f"{bpc:.4f}")

            console.print("\n")
            console.print(table)
        return

    # Wrap in UltronForCausalLM
    hf_config = UltronHFConfig.from_ultron_config(native_config)
    hf_model = UltronForCausalLM(hf_config)
    hf_model.load_state_dict(native_model.state_dict())
    hf_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Wrap in HFLM for in-memory evaluation
    lm_eval_model = HFLM(
        pretrained=hf_model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=str(accelerator.device),
    )

    if accelerator.is_main_process:
        console.print(f"[bold blue]📊 Running online lm-eval benchmarks for tasks:[/bold blue] [bold yellow]{task_list}[/bold yellow]")
        if args.limit:
            console.print(f"[bold yellow]⚠️ Sample limit per task: {args.limit}[/bold yellow]")

    # 3. Run evaluation
    results = simple_evaluate(
        model=lm_eval_model,
        tasks=task_list,
        limit=args.limit,
    )

    # 4. Display rich summary table
    if accelerator.is_main_process and results and "results" in results:
        table = Table(title="🏆 Ultron Benchmark Results (lm-eval)", show_header=True, header_style="bold magenta")
        table.add_column("Task", style="cyan")
        table.add_column("Metric", style="yellow")
        table.add_column("Score", style="bold green")

        for task_name, task_results in results["results"].items():
            for metric, val in task_results.items():
                if isinstance(val, (int, float)) and not metric.endswith("_stderr"):
                    table.add_row(task_name, metric, f"{val:.4f}")

        console.print("\n")
        console.print(table)


if __name__ == "__main__":
    main()
