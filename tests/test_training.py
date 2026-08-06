"""CPU-safe training-loop and checkpoint regression tests."""

from contextlib import nullcontext
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from train import build_config
from trainer import UltronTrainer


class TinyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)
        self.head = torch.nn.Linear(8, 16)

    def forward(self, inputs, targets=None):
        logits = self.head(self.embedding(inputs))
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                targets.flatten(),
            )
        return SimpleNamespace(logits=logits, loss=loss)


class FakeAccelerator:
    gradient_accumulation_steps = 1
    sync_gradients = True
    device = torch.device("cpu")
    is_main_process = True

    def __init__(self):
        self.skipped_batches = None
        self.saved = 0

    def accumulate(self, _model):
        return nullcontext()

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def reduce(self, tensor, reduction):
        assert reduction == "sum"
        return tensor

    def skip_first_batches(self, dataloader, count):
        self.skipped_batches = count
        return list(dataloader)[count:]

    def save_state(self, directory):
        self.saved += 1
        os.makedirs(directory, exist_ok=True)

    def wait_for_everyone(self):
        pass

    def print(self, _message):
        pass


class FakeTelemetry:
    def __init__(self):
        self.evaluations = []

    def print_message(self, _message):
        pass

    def update_terminal_progress(self, _step, loss):
        return 0

    def log_training_step(self, **_kwargs):
        pass

    def log_evaluation(self, step, train_loss, dev_loss, lr, eta_seconds):
        self.evaluations.append((step, train_loss, dev_loss, lr, eta_seconds))

    def get_wandb_run_id(self):
        return None

    def close(self):
        pass


def make_trainer(max_steps=2, is_test_mode=True):
    inputs = torch.randint(0, 16, (4, 6))
    targets = torch.roll(inputs, shifts=-1, dims=1)
    dataloader = DataLoader(TensorDataset(inputs, targets), batch_size=1)
    config = SimpleNamespace(
        B=1,
        T=6,
        max_steps=max_steps,
        warmup_steps=1,
        learning_rate=1e-2,
        min_lr=1e-3,
        eval_interval=1,
        is_test_mode=is_test_mode,
    )
    model = TinyLanguageModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    accelerator = FakeAccelerator()
    trainer = UltronTrainer(
        model,
        None,
        optimizer,
        dataloader,
        dataloader,
        config,
        accelerator,
    )
    trainer.telemetry = FakeTelemetry()
    return trainer


def test_training_pipeline_reaches_evaluation_without_checkpointing():
    trainer = make_trainer()

    trainer.train()

    assert trainer.step == 2
    assert len(trainer.telemetry.evaluations) == 2
    assert trainer.accelerator.saved == 0


def test_resume_skips_consumed_batches():
    trainer = make_trainer(max_steps=2)
    trainer.step = 1

    trainer.train()

    assert trainer.accelerator.skipped_batches == 1
    assert trainer.step == 2


def test_custom_test_length_remains_checkpoint_safe():
    config = build_config(SimpleNamespace(mode="test", max_steps=7))

    assert config.is_test_mode is True
    assert config.max_steps == 7


def test_only_main_process_writes_checkpoint_metadata(tmp_path):
    trainer = make_trainer(is_test_mode=False)
    trainer.accelerate_dir = str(tmp_path / "checkpoint")
    trainer.accelerator.is_main_process = False

    trainer.save_checkpoint()

    state_file = tmp_path / "checkpoint" / "training_state.json"
    assert not state_file.exists()

    trainer.accelerator.is_main_process = True
    trainer.save_checkpoint()

    assert state_file.exists()
