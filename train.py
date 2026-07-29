"""
GPT-2 (124M) Pre-training Script (2026 SOTA)

Usage:
    accelerate launch train.py [--mode=fresh|continue|test] [--max-steps=N]
"""

import argparse
import dataclasses
import os
import torch
from accelerate import Accelerator


from config import GPT2Config
from model import GPT2
from dataset import get_dataloaders




def print_rich(accelerator: Accelerator, text: str):
    pass


def main():
    parser = argparse.ArgumentParser(description="GPT-2 (124M) Pre-training")
    parser.add_argument("--mode", type=str, choices=["fresh", "continue", "test"], default="continue", help="Training execution mode: 'fresh', 'continue' (default), or 'test'")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional max training steps override (e.g. --max-steps=1000)")
    args = parser.parse_args()

    if not any(k in os.environ for k in ["ACCELERATE_TORCH_DEVICE", "ACCELERATE_PROCESS_ID", "LOCAL_RANK", "ACCELERATE_MIXED_PRECISION"]):
        raise RuntimeError("train.py must be launched using HuggingFace Accelerate!\nRun: accelerate launch train.py [--mode=fresh|continue|test] [--max-steps=N]")

    # Build configuration dynamically
    config = GPT2Config.from_metadata()
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    elif args.mode == "test":
        config.max_steps = 100
        config.is_test_mode = True

    accelerator = Accelerator(gradient_accumulation_steps=config.grad_accum_steps, log_with="wandb")
    accelerator.init_trackers("gpt2-pretraining", config=dataclasses.asdict(config))

    train_loader, dev_loader, _ = get_dataloaders(config, accelerator)

    model = GPT2(config)
    torch.set_float32_matmul_precision('high')

    # Muon (2D weight matrices) + AdamW (embeddings, norms, biases)
    optimizer_muon, optimizer_adamw = model.configure_optimizers(config.learning_rate)

    model = torch.compile(model)
    model, optimizer_muon, optimizer_adamw = accelerator.prepare(model, optimizer_muon, optimizer_adamw)

    from trainer import GPT2Trainer
    trainer = GPT2Trainer(model, optimizer_muon, optimizer_adamw, train_loader, dev_loader, config, accelerator)

    # Load checkpoint depending on --mode=fresh|continue|test
    if args.mode == "test":
        pass
    elif args.mode == "fresh":
        pass
    elif args.mode == "continue":
        # print_rich removed
        trainer.load_checkpoint()

    trainer.train()
    accelerator.end_training()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
