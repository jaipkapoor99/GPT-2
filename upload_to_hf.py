"""
Hugging Face Model Hub Upload Script
Uploads the trained GPT-2 (124M) SOTA model folder, configurations, weights, and codebase to Hugging Face Model Hub.
"""

import os
from huggingface_hub import HfApi

def clean_safetensors(file_path: str):
    if not os.path.exists(file_path):
        return
    from safetensors.torch import load_file, save_file
    state = load_file(file_path)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        print(f"Cleaning '_orig_mod.' prefixes from {file_path} before upload...")
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        save_file(clean_state, file_path)
        print("✓ Successfully cleaned safetensors keys.")

def main():
    repo_id = "jaipkapoor99/gpt2-2026-sota"
    local_dir = "/home/jaipkapoor99/Kaggle/Andrej Karpathy Course/GPT-2"
    model_dir = os.path.join(local_dir, "gpt2-fineweb-124m")
    
    print(f"=== HUGGING FACE MODEL HUB UPLOAD ===")
    print(f"Model Directory: {model_dir}")
    print(f"Target Repository: https://huggingface.co/{repo_id}\n")
    
    # Clean model.safetensors keys before upload
    clean_safetensors(os.path.join(model_dir, "model.safetensors"))
    
    # Ensure unbloated tokenizer files are included in the model directory for pipeline compatibility
    print("Copying unbloated tokenizer files to model directory...")
    from huggingface_hub import hf_hub_download
    import shutil
    for fn in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = hf_hub_download(repo_id="HuggingFaceTB/SmolLM-135M", filename=fn)
        shutil.copyfile(src, os.path.join(model_dir, fn))
    print("✓ Copied unbloated tokenizer files.")
    
    api = HfApi()
    
    # 1. Upload model directory (config.json, model.safetensors, remote code modules)
    print("1. Uploading model weights, configurations, and remote code modules...")
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload SOTA GPT-2 model weights, config, and remote code modules"
    )
    
    # 2. Upload Model Card README
    print("\n2. Uploading Model Card README...")
    api.upload_file(
        path_or_fileobj=os.path.join(local_dir, "README.md"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload model card README.md"
    )
    
    print(f"\n✓ UPLOAD COMPLETE!")
    print(f"Your weights and model metadata are live at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
