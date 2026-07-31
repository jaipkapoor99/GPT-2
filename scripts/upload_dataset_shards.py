"""
Ultron FineWeb-Edu Dataset Shards Uploader Script

Uploads the 100 binary dataset shards (`shards_edu/*.bin` and `*_meta.json`)
to Hugging Face Datasets Hub (`jaipkapoor99/ultron-fineweb-edu-shards`).

Usage:
    python3 scripts/upload_dataset_shards.py [--repo-id=USER/REPO] [--private]
"""

import os
import glob
import argparse
from rich.console import Console
from huggingface_hub import HfApi, login

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UltronConfig

console = Console()

def main():
    config = UltronConfig()
    parser = argparse.ArgumentParser(description="Upload Ultron FineWeb-Edu binary dataset shards to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default=config.hf_dataset_repo_id, help=f"Target Hugging Face Dataset Repo ID (default: {config.hf_dataset_repo_id})")
    parser.add_argument("--shards-dir", type=str, default="shards_edu", help="Path to local shards directory")
    parser.add_argument("--private", action="store_true", help="Set target dataset repository to private")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)

    api = HfApi(token=token)

    if not os.path.exists(args.shards_dir):
        console.print(f"[bold red]❌ Error: Shards directory '{args.shards_dir}' does not exist![/bold red]")
        return

    bin_files = sorted(glob.glob(os.path.join(args.shards_dir, "*.bin")))
    meta_files = sorted(glob.glob(os.path.join(args.shards_dir, "*_meta.json")))
    
    console.print(f"[bold cyan]🤗 Target Hugging Face Dataset Repository:[/bold cyan] [bold white]{args.repo_id}[/bold white]")
    console.print(f"Found [bold yellow]{len(bin_files)}[/bold yellow] binary shards (.bin) and [bold yellow]{len(meta_files)}[/bold yellow] metadata files (.json) (~20 GB total).")

    api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True, private=args.private)

    console.print(f"[bold blue]🚀 Uploading '{args.shards_dir}' folder to Hugging Face Datasets Hub...[/bold blue]")
    api.upload_folder(
        folder_path=args.shards_dir,
        path_in_repo="shards_edu",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Upload 100 binary FineWeb-Edu dataset shards (10B tokens, uint16, SmolLM BPE)"
    )
    console.print(f"[bold green]🎉 All dataset shards uploaded successfully to https://huggingface.co/datasets/{args.repo_id} ![/bold green]")

if __name__ == "__main__":
    main()
