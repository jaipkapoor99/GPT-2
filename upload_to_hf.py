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
    
    print("\nUploading model checkpoint, code modules, and documentation to Hugging Face Hub...")
    folder_path = "/home/jaipkapoor99/Kaggle/Andrej Karpathy Course/GPT-2"
    
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=[
            "shards/*",
            "fineweb_tokens.bin",
            "*.pt",
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            ".ipynb_checkpoints/*"
        ]
    )
    
    print(f"\n✓ UPLOAD COMPLETE!")
    print(f"Your model and codebase are live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
