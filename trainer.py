import os
import json
import csv
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from accelerate import Accelerator

def clean_safetensors(file_path: str):
    if not os.path.exists(file_path):
        return
    from safetensors.torch import load_file, save_file
    state = load_file(file_path)
    if any(k.startswith("_orig_mod.") for k in state.keys()):
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        save_file(clean_state, file_path)

class GPT2Trainer:
    def __init__(self, model, optimizer_muon, optimizer_adamw, train_loader, dev_loader, config, accelerator: Accelerator):
        self.model = model
        self.optimizer_muon = optimizer_muon
        self.optimizer_adamw = optimizer_adamw
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.accelerator = accelerator
        
        self.history = []
        self.accelerate_dir = "accelerate_checkpoint"
        self.step = 0
        self.decay_start_step = int(0.8 * config.max_steps)
        self.tokens_per_step = (config.B * accelerator.gradient_accumulation_steps) * config.T

    def print_table_header(self):
        self.accelerator.print("+-------+------------+-----------+-----------+")
        self.accelerator.print("| Step  | Train Loss | Dev Loss  |    LR     |")
        self.accelerator.print("+-------+------------+-----------+-----------+")

    def print_table_row(self, step, train_loss, dev_loss, lr):
        self.accelerator.print(f"| {step:5d} |   {train_loss:7.4f}  |  {dev_loss:7.4f}  | {lr:8.2e}  |")

    def print_table_footer(self):
        self.accelerator.print("+-------+------------+-----------+-----------+")

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

    def load_checkpoint(self, resume, load_checkpoint_path):
        if resume or load_checkpoint_path:
            dir_to_load = load_checkpoint_path if (load_checkpoint_path and os.path.isdir(load_checkpoint_path)) else self.accelerate_dir
            if os.path.isdir(dir_to_load):
                self.accelerator.print(f"Resuming full Accelerate training state from '{dir_to_load}'...")
                self.accelerator.load_state(dir_to_load)
                
                if os.path.exists(os.path.join(dir_to_load, "training_state.json")):
                    with open(os.path.join(dir_to_load, "training_state.json"), "r") as f:
                        state_info = json.load(f)
                        self.step = state_info.get("step", 0)
                        self.accelerator.print(f"✓ Restored training step: {self.step:,}")
                if os.path.exists("loss_history.json"):
                    try:
                        with open("loss_history.json", "r") as f:
                            self.history = json.load(f)
                        self.accelerator.print(f"✓ Restored loss history ({len(self.history)} previous evaluations)")
                    except Exception:
                        pass
                self.accelerator.print(f"✓ State restored successfully!")

    def evaluate(self, train_loss, lr):
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
        self.accelerator.log({"dev_loss": avg_dev_loss}, step=self.step)
        
        self.history.append({
            "step": self.step,
            "train_loss": train_loss,
            "dev_loss": avg_dev_loss,
            "learning_rate": lr
        })
        self.save_metrics()
        self.plot_loss_curve()
        self.save_checkpoint()
        self.model.train()

    def save_checkpoint(self, final=False):
        self.accelerator.save_state(self.accelerate_dir)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped_model, "_orig_mod"):
            unwrapped_model = unwrapped_model._orig_mod
        self.accelerator.save_model(unwrapped_model, "gpt2-fineweb-124m")
        if self.accelerator.is_main_process:
            clean_safetensors(os.path.join("gpt2-fineweb-124m", "model.safetensors"))
            with open(os.path.join(self.accelerate_dir, "training_state.json"), "w") as f:
                json.dump({"step": self.step, "max_steps": self.config.max_steps}, f, indent=2)
            if final:
                self.accelerator.print("✓ Saved full Accelerate state and model.")

    def save_metrics(self):
        if self.accelerator.is_main_process:
            with open("loss_history.json", "w") as f:
                json.dump(self.history, f, indent=2)
            with open("loss_history.csv", "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "dev_loss", "learning_rate"])
                writer.writeheader()
                writer.writerows(self.history)

    def plot_loss_curve(self):
        if self.accelerator.is_main_process and len(self.history) > 0:
            steps_list = [h["step"] for h in self.history]
            train_list = [h["train_loss"] for h in self.history]
            dev_list   = [h["dev_loss"] for h in self.history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(steps_list, train_list, label="Train Loss", marker="o", linewidth=2)
            plt.plot(steps_list, dev_list, label="Dev Loss", marker="s", linewidth=2)
            plt.xlabel("Step", fontsize=12)
            plt.ylabel("Cross Entropy Loss", fontsize=12)
            plt.title("GPT-2 (124M) Pre-training Loss Curve", fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig("loss_curve.svg", format="svg")
            plt.close()
            self.accelerator.print("✓ Vector loss curve updated dynamically: 'loss_curve.svg'")

    def train(self):
        self.model.train()
        
        self.accelerator.print(f"Starting pre-training for {self.config.max_steps} steps ({self.tokens_per_step * self.config.max_steps:,} total tokens)...\n")
        self.print_table_header()
        
        pbar = tqdm(total=self.config.max_steps, initial=self.step, desc="Pre-training GPT-2")
        
        if self.step > 0:
            skip_count = (self.step * self.accelerator.gradient_accumulation_steps) % len(self.train_loader)
            self.accelerator.print(f"Fast-forwarding data loader past {skip_count:,} batches in current epoch...")
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
                    pbar.update(1)
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
                    self.accelerator.log({"train_loss": loss.item(), "lr": lr}, step=self.step)
                    
                    if self.step % self.config.eval_interval == 0 or self.step == self.config.max_steps:
                        self.evaluate(loss.item(), lr)

        pbar.close()
        self.print_table_footer()
        self.accelerator.print("\nPre-training Complete!")
        self.save_checkpoint(final=True)
