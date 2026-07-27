"""
Hugging Face Model Hub Upload Script
Uploads the trained GPT-2 (124M) SOTA checkpoint and codebase to Hugging Face Model Hub.
"""

import os
from huggingface_hub import HfApi

def main():
    repo_id = "jaipkapoor99/gpt2-2026-sota"
    print(f"=== HUGGING FACE MODEL HUB UPLOAD ===")
    print(f"Target Repository: https://huggingface.co/{repo_id}")
    
    api = HfApi()
    
    print("\nCreating Hugging Face model repository if it doesn't exist...")
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    
    folder_path = "/home/jaipkapoor99/Kaggle/Andrej Karpathy Course/GPT-2"
    safetensors_dir = os.path.join(folder_path, "gpt2-fineweb-124m")
    
    print("\n1. Uploading code modules, scripts, and documentation...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "shards/*",
            "fineweb_tokens.bin",
            "gpt2-fineweb-124m/*",
            "*.bin",
            "*.pt",
            "*.pth",
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            ".ipynb_checkpoints/*"
        ]
    )

    print("\n2. Uploading Safetensors weights & HF Transformers config files...")
    safetensors_file = os.path.join(safetensors_dir, "model.safetensors")
    config_file = os.path.join(safetensors_dir, "config.json")
    gen_config_file = os.path.join(safetensors_dir, "generation_config.json")
    
    if os.path.exists(safetensors_file):
        api.upload_file(
            path_or_fileobj=safetensors_file,
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
            commit_message="Update trained SOTA GPT-2 model weights"
        )
    if os.path.exists(config_file):
        api.upload_file(
            path_or_fileobj=config_file,
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model"
        )
    if os.path.exists(gen_config_file):
        api.upload_file(
            path_or_fileobj=gen_config_file,
            path_in_repo="generation_config.json",
            repo_id=repo_id,
            repo_type="model"
        )
    
    print(f"\n✓ UPLOAD COMPLETE!")
    print(f"Your model (safetensors) and codebase are live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
