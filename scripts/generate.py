"""
generate.py — Text generation from a local Ultron Accelerate checkpoint.

Usage:
    accelerate launch generate.py
    accelerate launch generate.py --prompt "The theory of relativity" --max-tokens 150 --temperature 0.8 --top-k 50
"""

import argparse
import os
import json
import torch
from transformers import AutoTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accelerate import Accelerator
from config import UltronConfig
from model import UltronModel

# ── Accelerate guard ────────────────────────────────────────────────────────
if not any(k in os.environ for k in [
    "ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"
]):
    raise RuntimeError("Run with: accelerate launch generate.py ...")

# Initialize Accelerator
accelerator = Accelerator()


def load_model_weights(model: UltronModel, checkpoint_dir: str) -> UltronModel:
    """Load weights from an Accelerate checkpoint, stripping _orig_mod. prefix if present."""
    weight_file = None
    for fname in ("model.safetensors", "pytorch_model.bin",
                  "pytorch_model/model.safetensors", "pytorch_model/pytorch_model.bin"):
        candidate = os.path.join(checkpoint_dir, fname)
        if os.path.exists(candidate):
            weight_file = candidate
            break

    if weight_file is None:
        raise FileNotFoundError(
            f"No model weights found in '{checkpoint_dir}'.\n"
            "Expected one of: pytorch_model/model.safetensors, pytorch_model.bin, model.safetensors"
        )

    accelerator.print(f"Loading weights from: {weight_file}")

    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file, device=str(accelerator.device))
    else:
        state_dict = torch.load(weight_file, map_location=accelerator.device, weights_only=True)

    # Strip _orig_mod. prefix inserted by torch.compile
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.removeprefix("_orig_mod.")
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        accelerator.print(f"  Missing keys  : {len(missing)}")
    if unexpected:
        accelerator.print(f"  Unexpected    : {len(unexpected)}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Ultron text generation")
    parser.add_argument("--prompt", type=str, default="Hello, my name is",
                        help="Prompt to continue")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--checkpoint", type=str, default="accelerate_checkpoint",
                        help="Accelerate checkpoint directory")
    args = parser.parse_args()

    accelerator.print(f"Device : {accelerator.device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    config = UltronConfig()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    accelerator.print(f"Loading tokenizer ({config.tokenizer_name})...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    model = UltronModel(config)
    model = load_model_weights(model, args.checkpoint)

    # Prepare model with Accelerate for device/precision handling
    model = accelerator.prepare(model)
    model.eval()

    # Print step info if available
    state_file = os.path.join(args.checkpoint, "training_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            st = json.load(f)
        accelerator.print(f"Step   : {st['step']:,} / {st['max_steps']:,}")

    # ── Generate ──────────────────────────────────────────────────────────────
    accelerator.print(f"\nPrompt : {args.prompt!r}\n")

    # Encode input prompt and move to device
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt")
    input_ids = input_ids.to(accelerator.device)

    # Unwrap model for inference if wrapped in DDP/FSDP container
    unwrapped_model = accelerator.unwrap_model(model)

    with torch.no_grad(), accelerator.autocast():
        output_ids = unwrapped_model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    accelerator.print("=" * 70)
    accelerator.print(generated)
    accelerator.print("=" * 70)


if __name__ == "__main__":
    main()