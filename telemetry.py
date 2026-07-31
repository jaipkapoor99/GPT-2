"""
Ultron Telemetry Module (2026 SOTA)

Encapsulates W&B tracking, live terminal progress meters, ETA calculations,
metrics summary definitions, and checkpoint state telemetry resolution.
"""

import os
import json
import time
import shutil
import dataclasses
from typing import Dict, Any, Optional
from accelerate import Accelerator
from tqdm import tqdm

class UltronTelemetry:
    """Dedicated Telemetry & Experiment Tracking Manager."""

    def __init__(self, config, accelerator: Accelerator, checkpoint_dir: str = "accelerate_checkpoint"):
        self.config = config
        self.accelerator = accelerator
        self.checkpoint_dir = checkpoint_dir
        self.start_time: Optional[float] = None
        self.session_steps: int = 0
        self.pbar: Optional[tqdm] = None
        self.last_dev_loss: Optional[float] = None
    @classmethod
    def setup_accelerator_trackers(cls, config, args, checkpoint_dir: str = "accelerate_checkpoint") -> Accelerator:
        """Helper to configure Accelerator trackers with W&B run ID resumption."""
        is_test = (getattr(args, "mode", None) == "test") or getattr(config, "is_test_mode", False)
        
        if is_test:
            # Disable Weights & Biases telemetry completely on test runs
            accelerator = Accelerator(gradient_accumulation_steps=config.grad_accum_steps, log_with=None)
            return accelerator

        wandb_init_kwargs = {"wandb": {"name": "master"}}
        state_file = os.path.join(checkpoint_dir, "training_state.json")
        
        if getattr(args, "mode", "continue") == "fresh":
            wandb_init_kwargs["wandb"].update({"resume": "never"})
        elif getattr(args, "mode", "continue") == "continue" and os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    state_data = json.load(f)
                    if "wandb_run_id" in state_data:
                        wandb_init_kwargs["wandb"].update({"id": state_data["wandb_run_id"], "resume": "must"})
            except Exception:
                pass

        accelerator = Accelerator(gradient_accumulation_steps=config.grad_accum_steps, log_with="wandb")
        accelerator.init_trackers(
            "ultron-pretraining",
            config=dataclasses.asdict(config),
            init_kwargs=wandb_init_kwargs
        )

        if accelerator.is_main_process:
            try:
                import wandb
                # Structured metric definitions and summaries
                wandb.define_metric("step")
                wandb.define_metric("train/*", step_metric="step")
                wandb.define_metric("eval/*", step_metric="step")
                wandb.define_metric("perf/*", step_metric="step")

                wandb.define_metric("eval/dev_loss", summary="min")
                wandb.define_metric("train/train_loss", summary="last")
            except Exception:
                pass

        return accelerator

    def get_wandb_run_id(self) -> Optional[str]:
        """Extract the current active W&B run ID if available."""
        if not self.accelerator.is_main_process:
            return None
        try:
            wandb_tracker = self.accelerator.get_tracker("wandb")
            if wandb_tracker is not None and hasattr(wandb_tracker, "run"):
                return wandb_tracker.run.id
        except Exception:
            pass
        return None

    def _init_pbar(self, initial_step: int):
        """Initialize tqdm progress bar on main process."""
        if self.accelerator.is_main_process and self.pbar is None:
            self.pbar = tqdm(
                total=self.config.max_steps,
                initial=initial_step,
                desc="⚡ Pre-training",
                unit="step",
                dynamic_ncols=True,
                leave=True
            )

    def set_last_dev_loss(self, dev_loss: float):
        """Record the latest dev loss from evaluation."""
        self.last_dev_loss = dev_loss

    def update_terminal_progress(self, current_step: int, loss: Optional[float] = None) -> int:
        """Update live tqdm progress bar with throughput (tok/s), train loss, and dev loss."""
        if self.start_time is None:
            self.start_time = time.time()
            self.session_steps = 0

        self.session_steps += 1
        if self.pbar is None:
            self._init_pbar(current_step)

        elapsed = time.time() - self.start_time
        remaining_steps = max(0, self.config.max_steps - current_step)
        steps_per_sec = self.session_steps / elapsed if elapsed > 0 else 0.0
        tokens_per_step = (self.config.B * self.accelerator.gradient_accumulation_steps) * self.config.T
        tokens_per_sec = steps_per_sec * tokens_per_step
        self.last_throughput = tokens_per_sec
        self.last_steps_per_sec = steps_per_sec
        eta = (remaining_steps / steps_per_sec) if steps_per_sec > 0 else 0

        if self.accelerator.is_main_process and self.pbar is not None:
            # Update tqdm progress bar position and postfix stats
            self.pbar.n = current_step
            
            if tokens_per_sec >= 1e6:
                tok_str = f"{tokens_per_sec / 1e6:.2f}M tok/s"
            elif tokens_per_sec >= 1e3:
                tok_str = f"{tokens_per_sec / 1e3:.1f}k tok/s"
            else:
                tok_str = f"{tokens_per_sec:.0f} tok/s"
                
            postfix_dict = {"tok/s": tok_str}
            if loss is not None:
                postfix_dict["train_loss"] = f"{loss:.4f}"
            if self.last_dev_loss is not None:
                postfix_dict["dev_loss"] = f"{self.last_dev_loss:.4f}"
                
            self.pbar.set_postfix(postfix_dict, refresh=True)

        return int(eta)

    def print_message(self, text: str):
        """Print log message cleanly above tqdm progress bar."""
        if self.accelerator.is_main_process:
            if self.pbar is not None:
                self.pbar.write(text)
            else:
                print(text)

    def close(self):
        """Close tqdm progress bar safely."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None

    def log_step(self, metrics: Dict[str, Any], step: int):
        """Log metric key-values to Accelerate trackers."""
        self.accelerator.log(metrics, step=step)

    def log_training_step(self, step: int, loss: float, lr: float, progress_percent: float, eta_seconds: int):
        """Log structured training metrics per iteration step."""
        tok_per_sec = getattr(self, "last_throughput", 0.0)
        self.log_step({
            "train/loss": loss,
            "train/lr": lr,
            "train/progress_percent": progress_percent,
            "perf/tokens_per_sec": tok_per_sec,
            "perf/steps_per_sec": getattr(self, "last_steps_per_sec", 0.0),
            "perf/eta_seconds": eta_seconds,
            # Top-level keys for W&B unified line graph & backwards compatibility
            "train_loss": loss,
            "lr": lr,
            "tokens_per_sec": tok_per_sec,
            "progress_percent": progress_percent,
            "eta_seconds": eta_seconds,
        }, step=step)

    def log_evaluation(self, step: int, train_loss: float, dev_loss: float, lr: float, eta_seconds: int):
        """Log evaluation metrics at validation steps (train_loss and dev_loss logged together on same step)."""
        self.set_last_dev_loss(dev_loss)
        tok_per_sec = getattr(self, "last_throughput", 0.0)
        self.log_step({
            "eval/dev_loss": dev_loss,
            "train/loss": train_loss,
            "train/lr": lr,
            "perf/tokens_per_sec": tok_per_sec,
            "perf/steps_per_sec": getattr(self, "last_steps_per_sec", 0.0),
            "perf/eta_seconds": eta_seconds,
            # Top-level keys for W&B side-by-side train & dev loss chart
            "dev_loss": dev_loss,
            "train_loss": train_loss,
            "lr": lr,
            "tokens_per_sec": tok_per_sec,
            "eta_seconds": eta_seconds,
        }, step=step)


class TokenizationTelemetry:
    """Dedicated Telemetry Manager for Tokenization & Data Sharding using tqdm."""

    def __init__(self, target_tokens: int, start_tokens: int = 0):
        self.target_tokens = target_tokens
        self.start_tokens = start_tokens
        self.start_time: Optional[float] = None
        self.session_tokens_processed: int = 0
        self.pbar: Optional[tqdm] = tqdm(
            total=target_tokens,
            initial=start_tokens,
            desc="📦 Tokenizing",
            unit="tok",
            unit_scale=True,
            dynamic_ncols=True,
            leave=True
        )

    def update(self, added_tokens: int, current_total: int, shard_info: Optional[str] = None):
        """Update live tqdm tokenization progress, speed, and ETA."""
        if self.start_time is None:
            self.start_time = time.time()

        self.session_tokens_processed += added_tokens
        elapsed = time.time() - self.start_time
        tok_per_sec = self.session_tokens_processed / elapsed if elapsed > 0 else 0.0

        if tok_per_sec >= 1e6:
            tok_str = f"{tok_per_sec / 1e6:.2f}M tok/s"
        elif tok_per_sec >= 1e3:
            tok_str = f"{tok_per_sec / 1e3:.1f}k tok/s"
        else:
            tok_str = f"{tok_per_sec:.0f} tok/s"

        if self.pbar is not None:
            self.pbar.n = current_total
            postfix = {"tok/s": tok_str}
            if shard_info:
                postfix["shard"] = shard_info
            self.pbar.set_postfix(postfix, refresh=True)

    def print_message(self, text: str):
        """Print clean log message above tqdm bar."""
        if self.pbar is not None:
            self.pbar.write(text)
        else:
            print(text)

    def close(self):
        """Close tqdm progress bar cleanly."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
