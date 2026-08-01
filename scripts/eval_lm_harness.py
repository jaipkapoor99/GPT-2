"""
scripts/eval_lm_harness.py — Official lm-evaluation-harness Evaluator for Ultron (124M)

Evaluates the Ultron (124M) base checkpoint using EleutherAI's `lm-evaluation-harness` across:
- `arc_easy`
- `hellaswag`
- `mmlu`

Usage:
    accelerate launch scripts/eval_lm_harness.py [--tasks=arc_easy,hellaswag] [--limit=50]
"""

import os
import sys
import json
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator
from config import UltronConfig
from model import UltronModel
from transformers import AutoTokenizer
import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate

if not any(k in os.environ for k in ["ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"]):
    raise RuntimeError("Run with: accelerate launch scripts/eval_lm_harness.py ...")

accelerator = Accelerator()


def load_base_model(checkpoint_dir: str, config: UltronConfig) -> UltronModel:
    """Instantiate model and load weights from Accelerate checkpoint."""
    model = UltronModel(config)
    weight_file = None
    for fname in ("model.safetensors", "pytorch_model.bin", "pytorch_model/model.safetensors", "pytorch_model/pytorch_model.bin"):
        candidate = os.path.join(checkpoint_dir, fname)
        if os.path.exists(candidate):
            weight_file = candidate
            break

    if weight_file is None:
        raise FileNotFoundError(f"No checkpoint weights found in '{checkpoint_dir}'")

    accelerator.print(f"Loading weights from: {weight_file}")
    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file, device=str(accelerator.device))
    else:
        state_dict = torch.load(weight_file, map_location=accelerator.device, weights_only=True)

    cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(accelerator.device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Ultron EleutherAI lm-evaluation-harness Suite")
    parser.add_argument("--checkpoint-dir", type=str, default="accelerate_checkpoint", help="Path to checkpoint directory")
    parser.add_argument("--tasks", type=str, default="arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande",
                        help="Comma-separated evaluation tasks (e.g., arc_easy,arc_challenge,hellaswag,openbookqa,piqa,winogrande,mmlu)")
    parser.add_argument("--limit", type=int, default=50, help="Number of benchmark samples per evaluation task")
    args = parser.parse_args()

    config = UltronConfig()
    model = load_base_model(args.checkpoint_dir, config)

    accelerator.print(f"Loading tokenizer ({config.tokenizer_name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Wrap model with lm-eval HFLM wrapper
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1,
        device=str(accelerator.device)
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    eval_limit = args.limit if args.limit and args.limit > 0 else None
    accelerator.print(f"\n🚀 Running EleutherAI lm-evaluation-harness across tasks: {task_list} (Limit: {eval_limit if eval_limit else 'FULL'})...\n")

    results = simple_evaluate(
        model=lm_eval_model,
        tasks=task_list,
        limit=eval_limit
    )

    accelerator.print("\n==================================================")
    accelerator.print("📊 ELEUTHERAI LM-EVALUATION-HARNESS REPORT")
    accelerator.print("==================================================")
    
    if "results" in results:
        for task_name, task_metrics in results["results"].items():
            acc = task_metrics.get("acc,none") or task_metrics.get("acc_norm,none") or task_metrics.get("acc")
            if acc is not None:
                accelerator.print(f"• {task_name:<25} : {acc*100:.2f}%")
            else:
                accelerator.print(f"• {task_name:<25} : {task_metrics}")
    accelerator.print("==================================================\n")

    os.makedirs("logs", exist_ok=True)
    out_path = os.path.join("logs", "pre_training_checkpoint_eval.json")
    with open(out_path, "w") as f:
        # Convert non-serializable objects
        json.dump(results.get("results", {}), f, indent=2, default=str)

    accelerator.print(f"✓ Results saved to '{out_path}'")


if __name__ == "__main__":
    main()
