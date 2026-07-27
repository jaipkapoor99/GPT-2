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
    api.upload_file(
        path_or_fileobj=os.path.join(safetensors_dir, "model.safetensors"),
        path_in_repo="model.safetensors",
        repo_id=repo_id,
        repo_type="model"
    )
    api.upload_file(
        path_or_fileobj=os.path.join(safetensors_dir, "config.json"),
        path_in_repo="config.json",
        repo_id=repo_id,
        repo_type="model"
    )
    api.upload_file(
        path_or_fileobj=os.path.join(safetensors_dir, "generation_config.json"),
        path_in_repo="generation_config.json",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"\n✓ UPLOAD COMPLETE!")
    print(f"Your model (safetensors) and codebase are live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
