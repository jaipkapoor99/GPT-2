"""
Hugging Face Model Hub Upload Script
Uploads the trained GPT-2 (124M) SOTA model folder, configurations, weights, and codebase to Hugging Face Model Hub.
"""

import os
from huggingface_hub import HfApi

def main():
    repo_id = "jaipkapoor99/gpt2-2026-sota"
    local_dir = "/home/jaipkapoor99/Kaggle/Andrej Karpathy Course/GPT-2"
    model_dir = os.path.join(local_dir, "gpt2-fineweb-124m")
    
    print(f"=== HUGGING FACE MODEL HUB UPLOAD ===")
    print(f"Model Directory: {model_dir}")
    print(f"Target Repository: https://huggingface.co/{repo_id}\n")
    
    api = HfApi()
    
    # 1. Upload model directory (config.json, model.safetensors, remote code modules)
    print("1. Uploading model weights, configurations, and remote code modules...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload SOTA GPT-2 model weights, config, and remote code modules"
    )
    
    # 2. Upload codebase scripts and documentation
    print("\n2. Uploading codebase scripts, test scripts, and documentation...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "shards/*",
            "fineweb_tokens.bin",
            "gpt2-fineweb-124m/*",
            "clean_model.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            ".ipynb_checkpoints/*"
        ],
        commit_message="Upload project scripts and documentation"
    )
    
    print(f"\n✓ UPLOAD COMPLETE!")
    print(f"Your model and codebase are live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
