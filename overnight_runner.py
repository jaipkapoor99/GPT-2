"""
Overnight Continuous Training Supervisor Script
Monitors active pre-training run. When step 70,000 finishes, seamlessly resumes
training across 100% of the FineWeb dataset (152,587 total steps = 10.0 Billion Tokens)
and uploads final model weights to Hugging Face Model Hub upon completion.
"""

import os
import time
import subprocess
from huggingface_hub import HfApi

def is_train_running():
    try:
        output = subprocess.check_output(["pgrep", "-f", "train.py"]).decode()
        return len(output.strip()) > 0
    except Exception:
        return False

def main():
    print("=== OVERNIGHT FULL-DATASET SUPERVISOR LAUNCHED ===")
    print("Target: 152,587 steps (10.00 Billion Tokens - 100% FineWeb Dataset)")
    
    # 1. Wait for current 70k step run to complete
    while is_train_running():
        time.sleep(30)
        
    print("\nCurrent run finished! Resuming pre-training for full 10B tokens (152,587 steps)...")
    
    # 2. Launch resumption to 152,587 steps
    cmd = ["python", "train.py", "--resume", "--optimizer", "muon", "--max-steps", "152587"]
    with open("train.log", "a") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
        proc.wait()
        
    print("\n✓ Full 10 Billion Token Pre-training Complete!")
    print("Uploading final model weights & artifacts to Hugging Face Model Hub...")
    
    # 3. Upload to Hugging Face Model Hub upon completion
    try:
        subprocess.run(["python", "upload_to_hf.py"], check=True)
        print("✓ Successfully uploaded final model to Hugging Face Model Hub!")
    except Exception as e:
        print(f"Upload error: {e}")

if __name__ == "__main__":
    main()
