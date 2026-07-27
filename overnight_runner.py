"""
Overnight Continuous Training Supervisor Script (Airtight Edition)
Monitors active pre-training run. When step 70,000 finishes, seamlessly resumes
training across 100% of the FineWeb dataset (152,587 total steps = 10.0 Billion Tokens)
and uploads final model weights to Hugging Face Model Hub upon completion.
"""

import os
import sys
import time
import json
import subprocess

# Ensure unbuffered output so logs appear in overnight.log immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

def log(msg: str):
    print(msg, flush=True)

def is_train_running() -> bool:
    """
    Airtight process check: searches specifically for python/accelerate processes running train.py.
    Excludes grep, tail, editors, and this supervisor process itself.
    Includes a 10-second confirmation debounce to avoid false negatives during checkpointing/restarts.
    """
    def _check_once():
        try:
            # Match python or accelerate processes executing train.py
            output = subprocess.check_output(["pgrep", "-f", "(python|accelerate).*train\\.py"]).decode()
            pids = [int(p) for p in output.strip().split() if p.isdigit()]
            # Exclude self PID and parent PID
            pids = [p for p in pids if p != os.getpid() and p != os.getppid()]
            return len(pids) > 0
        except Exception:
            return False

    if _check_once():
        return True
    # If false, wait 10 seconds and verify again to prevent transient false negatives
    time.sleep(10)
    return _check_once()

def get_current_step() -> int:
    state_file = os.path.join("accelerate_checkpoint", "training_state.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                data = json.load(f)
                return data.get("step", 0)
        except Exception:
            pass
    return 0

def run_with_retries(cmd: list, max_retries: int = 3, delay_secs: int = 30):
    for attempt in range(1, max_retries + 1):
        log(f"Executing: {' '.join(cmd)} (Attempt {attempt}/{max_retries})...")
        res = subprocess.run(cmd)
        if res.returncode == 0:
            log(f"✓ Successfully completed: {' '.join(cmd)}")
            return True
        log(f"⚠ Command failed with return code {res.returncode}. Retrying in {delay_secs} seconds...")
        time.sleep(delay_secs)
        delay_secs *= 2 # Exponential backoff
    raise RuntimeError(f"Command failed after {max_retries} attempts: {' '.join(cmd)}")

def main():
    log("=== OVERNIGHT FULL-DATASET SUPERVISOR LAUNCHED (AIRTIGHT EDITION) ===")
    log(f"Supervisor PID: {os.getpid()}")
    log("Target: 152,587 steps (10.00 Billion Tokens - 100% FineWeb Dataset)\n")
    
    # 1. Monitor current 70k step pre-training run
    log("1. Monitoring active 70k step pre-training run...")
    while is_train_running():
        step = get_current_step()
        log(f"   [Supervisor Status] Pre-training active. Current checkpoint step: {step:,} / 70,000")
        time.sleep(60)
        
    step = get_current_step()
    log(f"\n✓ Active pre-training process ended. Last recorded step: {step:,}.")
    
    if step >= 152500:
        log("Full 10 Billion token pre-training is already complete!")
    else:
        log("\n2. Launching pre-training resumption to full 10B tokens (152,587 steps)...")
        cmd = ["python", "-u", "train.py", "--resume", "--optimizer", "muon", "--max-steps", "152587"]
        with open("train.log", "a") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            ret = proc.wait()
            if ret != 0:
                log(f"❌ Resumption training failed with return code {ret}! Aborting supervisor.")
                sys.exit(ret)
                
        log("\n✓ Full 10 Billion Token Pre-training Complete!")
    
    # 3. Upload to Hugging Face Model Hub with automatic retries
    log("\n3. Uploading final model weights & codebase to Hugging Face Model Hub...")
    try:
        run_with_retries(["python", "-u", "upload_to_hf.py"], max_retries=3, delay_secs=30)
        log("✓ Successfully uploaded final model and codebase to Hugging Face Model Hub!")
    except Exception as e:
        log(f"❌ Upload error: {e}")
        sys.exit(1)
    
    log("\n=== OVERNIGHT SUPERVISOR SUCCESSFULLY COMPLETED ALL TASKS ===")

if __name__ == "__main__":
    main()
