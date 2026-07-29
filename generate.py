"""
generate.py — Text generation from a local GPT-2 Accelerate checkpoint.

Usage:
    python3 generate.py
    python3 generate.py --prompt "The theory of relativity" --max-tokens 150 --temperature 0.8 --top-k 50
"""

import argparse
import os
import torch
from transformers import AutoTokenizer

from config import GPT2Config
from model import GPT2


def load_model_weights(model: GPT2, checkpoint_dir: str, device: torch.device) -> GPT2:
    """Load weights from an Accelerate checkpoint, stripping _orig_mod. prefix if present."""
    # Accelerate saves model weights in a sub-folder named 'pytorch_model'
    # or directly as 'model.safetensors' / 'pytorch_model.bin'
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

    print(f"Loading weights from: {weight_file}")

    if weight_file.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weight_file, device=str(device))
    else:
        state_dict = torch.load(weight_file, map_location=device, weights_only=True)

    # Strip _orig_mod. prefix inserted by torch.compile
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.removeprefix("_orig_mod.")
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  Missing keys  : {len(missing)}")
    if unexpected:
        print(f"  Unexpected    : {len(unexpected)}")

    return model


def main():
    parser = argparse.ArgumentParser(description="GPT-2 text generation")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

    # ── Model ─────────────────────────────────────────────────────────────────
    config = GPT2Config()
    model = GPT2(config).to(device)
    model = load_model_weights(model, args.checkpoint, device)
    model.eval()

    # Print step info if available
    import json
    state_file = os.path.join(args.checkpoint, "training_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            st = json.load(f)
        print(f"Step   : {st['step']:,} / {st['max_steps']:,}")

    # ── Generate ──────────────────────────────────────────────────────────────
    print(f"\nPrompt : {args.prompt!r}\n")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("=" * 70)
    print(generated)
    print("=" * 70)


if __name__ == "__main__":
    main()
