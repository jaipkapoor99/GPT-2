"""
scripts/export_to_hf.py — Export local Accelerate checkpoint to Hugging Face format
and push to Hugging Face Hub (jaipkapoor99/ultron-124m).

Zero-flag CLI design. Runs unit verification tests before uploading.
"""

import os
import sys
import unittest
import torch
from huggingface_hub import HfApi

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import UltronConfig
from model import UltronModel
from hf_model import UltronHFConfig, UltronForCausalLM
from generate import load_model_weights

REPO_ID = "jaipkapoor99/ultron-124m"
CHECKPOINT_DIR = "accelerate_checkpoint"
EXPORT_DIR = "hf_export"


def run_unit_tests() -> bool:
    """Run tests/test_hf_wrapper.py before attempting export/upload."""
    print("🧪 Running pre-upload unit tests (tests/test_hf_wrapper.py)...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_hf_wrapper.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    if not any(k in os.environ for k in [
        "ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"
    ]):
        raise RuntimeError("Run with: accelerate launch scripts/export_to_hf.py ...")

    if not run_unit_tests():
        print("❌ Pre-upload unit tests failed! Aborting export and upload.")
        sys.exit(1)

    print(f"✓ Unit tests passed. Proceeding with export to '{EXPORT_DIR}'...")
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Instantiate model & config
    config = UltronConfig()
    hf_config = UltronHFConfig(
        vocab_size=config.vocab_size,
        n_positions=config.T,
        n_embd=config.C,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_kv_head=config.n_kv_head,
        dropout=config.dropout,
        rope_base=config.rope_base,
        logit_softcap=config.logit_softcap,
    )

    hf_model = UltronForCausalLM(hf_config)
    load_model_weights(hf_model, CHECKPOINT_DIR)

    # Save pretrained locally
    print(f"Saving HF format model to directory '{EXPORT_DIR}'...")
    hf_model.save_pretrained(EXPORT_DIR)
    hf_config.save_pretrained(EXPORT_DIR)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Copy hf_model.py, model.py, and config.py into export directory so trust_remote_code=True works out-of-the-box
    for fname in ("hf_model.py", "model.py", "config.py"):
        src = os.path.join(root_dir, fname)
        dst = os.path.join(EXPORT_DIR, fname)
        if os.path.exists(src):
            with open(src, "r") as f_src, open(dst, "w") as f_dst:
                f_dst.write(f_src.read())

    # Build model card README from model_card.yaml + root README.md
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    card_yaml_path = os.path.join(root_dir, "model_card.yaml")
    root_readme_path = os.path.join(root_dir, "README.md")
    export_readme_path = os.path.join(EXPORT_DIR, "README.md")

    yaml_content = ""
    if os.path.exists(card_yaml_path):
        with open(card_yaml_path, "r") as f:
            yaml_content = f.read().strip()

    readme_content = ""
    if os.path.exists(root_readme_path):
        with open(root_readme_path, "r") as f:
            readme_content = f.read().strip()

    combined_card = f"---\n{yaml_content}\n---\n\n{readme_content}\n"
    with open(export_readme_path, "w") as f:
        f.write(combined_card)

    print(f"Uploading exported files to Hugging Face Hub repository '{REPO_ID}'...")
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, exist_ok=True)
    api.upload_folder(
        folder_path=EXPORT_DIR,
        repo_id=REPO_ID,
        commit_message="Export Ultron 124M SOTA checkpoint to HF Hub",
    )
    print(f"🎉 Successfully exported and published Ultron (124M) to https://huggingface.co/{REPO_ID}!")


if __name__ == "__main__":
    main()
