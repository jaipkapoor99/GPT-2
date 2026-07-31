import os
import json
import shutil
import torch
import time
from accelerate import Accelerator
from telemetry import UltronTelemetry


class UltronTrainer:
    def __init__(self, model, optimizer_muon, optimizer_adamw, train_loader, dev_loader, config, accelerator: Accelerator):
        self.model = model
        self.optimizer_muon = optimizer_muon
        self.optimizer_adamw = optimizer_adamw
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.accelerator = accelerator
        
        self.accelerate_dir = "accelerate_checkpoint"
        self.step = 0
        self.decay_start_step = int(0.8 * config.max_steps)
        self.tokens_per_step = (config.B * accelerator.gradient_accumulation_steps) * config.T
        self.telemetry = UltronTelemetry(config, accelerator, checkpoint_dir=self.accelerate_dir)

    def print_rich(self, msg: str):
        if hasattr(self, "telemetry") and self.telemetry is not None:
            self.telemetry.print_message(msg)
        else:
            self.accelerator.print(msg)

    def print_table_header(self):
        pass

    def print_table_row(self, step, train_loss, dev_loss, lr):
        pass

    def print_table_footer(self):
        pass

    def update_learning_rate(self):
        # WSD (Warmup-Stable-Linear-Decay) Learning Rate Schedule
        if self.step < self.config.warmup_steps:
            lr = self.config.learning_rate * (self.step + 1) / self.config.warmup_steps
        elif self.step < self.decay_start_step:
            lr = self.config.learning_rate
        else:
            decay_ratio = (self.step - self.decay_start_step) / max(1, self.config.max_steps - self.decay_start_step)
            lr = self.config.min_lr + (1.0 - decay_ratio) * (self.config.learning_rate - self.config.min_lr)
            
        for param_group in self.optimizer_adamw.param_groups:
            param_group['lr'] = lr
            
        if self.optimizer_muon is not None:
            for param_group in self.optimizer_muon.param_groups:
                param_group['lr'] = 0.04 * (lr / self.config.learning_rate)
        return lr

    def load_checkpoint(self):
        if os.path.isdir(self.accelerate_dir):
            self.accelerator.print(f"Resuming training state from '{self.accelerate_dir}'...")
            self.accelerator.load_state(self.accelerate_dir)
            
            state_file = os.path.join(self.accelerate_dir, "training_state.json")
            if os.path.exists(state_file):
                with open(state_file, "r") as f:
                    state_info = json.load(f)
                    self.step = state_info.get("step", 0)
                    self.accelerator.print(f"✓ Restored training step: {self.step:,}")
            self.accelerator.print(f"✓ State restored successfully!")
        else:
            self.accelerator.print(f"⚠ No checkpoint found at '{self.accelerate_dir}', starting from scratch.")

    def evaluate(self, train_loss, lr, eta_seconds=0):
        self.model.eval()
        total_dev_loss = 0.0
        dev_batches = 0
        with torch.no_grad():
            for xb_dev, yb_dev in self.dev_loader:
                dev_out = self.model(xb_dev, yb_dev)
                dev_loss = dev_out.loss if (hasattr(dev_out, "loss") and dev_out.loss is not None) else dev_out[1]
                total_dev_loss += dev_loss.item()
                dev_batches += 1
                if dev_batches >= 20:
                    break
        avg_dev_loss = total_dev_loss / dev_batches

        self.print_table_row(self.step, train_loss, avg_dev_loss, lr)
        # Log train_loss and dev_loss together at the same step
        self.telemetry.log_evaluation(self.step, train_loss, avg_dev_loss, lr, eta_seconds=eta_seconds)
        self.save_checkpoint()
        self.model.train()

    def save_checkpoint(self, final=False):
        if getattr(self.config, "is_test_mode", False) or self.config.max_steps == 100:
            return
        # Save only the Accelerate training state (model weights, optimizer, etc.)
        self.accelerator.save_state(self.accelerate_dir)
        # Persist the current step and wandb run ID so we can resume correctly
        state_payload = {"step": self.step, "max_steps": self.config.max_steps}
        run_id = self.telemetry.get_wandb_run_id()
        if run_id:
            state_payload["wandb_run_id"] = run_id

        with open(os.path.join(self.accelerate_dir, "training_state.json"), "w") as f:
            json.dump(state_payload, f, indent=2)
        if final:
            self.print_rich("✓ Saved Accelerate checkpoint.")

    def train(self):
        self.model.train()
        
        self.print_rich(f"[bold yellow]⚡ Pre-training for {self.config.max_steps:,} steps ({self.tokens_per_step * self.config.max_steps:,} total tokens)...[/bold yellow]\n")
        # Training loop without tqdm progress bar
        
        if self.step > 0:
            skip_count = (self.step * self.accelerator.gradient_accumulation_steps) % len(self.train_loader)
            self.print_rich(f"[bold yellow]⏩ Fast-forwarding dataset past {skip_count:,} batches...[/bold yellow]")
            active_dataloader = self.train_loader
        else:
            active_dataloader = self.train_loader
            
        while self.step < self.config.max_steps:
            for xb, yb in active_dataloader:
                if self.step >= self.config.max_steps:
                    break
                    
                lr = self.update_learning_rate()
                    
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        out = self.model(xb, yb)
                        if hasattr(out, "loss") and out.loss is not None:
                            loss = out.loss
                        else:
                            loss = out[1]
                            
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                    self.optimizer_adamw.step()
                    if self.optimizer_muon is not None:
                        self.optimizer_muon.step()
                        
                    self.optimizer_adamw.zero_grad()
                    if self.optimizer_muon is not None:
                        self.optimizer_muon.zero_grad()
                    
                    if self.accelerator.sync_gradients:
                        self.step += 1
                        eta_seconds = self.telemetry.update_terminal_progress(self.step, loss=loss.item())
                        is_eval_step = (self.step % self.config.eval_interval == 0 or self.step == self.config.max_steps)
                        if not is_eval_step:
                            # Non-eval steps: log train metrics only
                            self.telemetry.log_training_step(
                                step=self.step,
                                loss=loss.item(),
                                lr=lr,
                                progress_percent=(self.step / self.config.max_steps) * 100,
                                eta_seconds=eta_seconds
                            )
                        if is_eval_step:
                            self.evaluate(loss.item(), lr, eta_seconds=eta_seconds)
        
        self.telemetry.close()
        self.print_rich("\n[bold green]🎉 Pre-training Complete![/bold green]")
        self.save_checkpoint(final=True)
