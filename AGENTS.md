# Repository Guidelines

## Project Structure & Module Organization

Ultron is a PyTorch decoder-only language-model project. Core architecture and configuration live in `model.py` and `config.py`. Training is split across `train.py`, `trainer.py`, `dataset.py`, and `telemetry.py`. Operational entry points belong in `scripts/`, including tokenization, generation, evaluation, and Hugging Face uploads. Tests live in `tests/`.

Large runtime artifacts are intentionally untracked: `shards_edu/`, `accelerate_checkpoint/`, `wandb/`, and `logs/`. Do not add model weights, token shards, credentials, or generated telemetry to commits.

## Build, Test, and Development Commands

```bash
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install torch==2.13.0
uv pip install -r requirements.lock
pytest -q
ULTRON_TEST_COMPILE=1 pytest -q tests/test_model.py -k torch_compile
accelerate launch train.py --mode=test
accelerate launch scripts/generate.py --prompt "Hello"
```

`requirements.lock` pins backend-neutral dependencies; install the appropriate PyTorch 2.13 wheel first. The standard test suite is CPU-safe. The compiler test is opt-in because it is slower and toolchain-dependent. `train.py --mode=test` exercises the training pipeline and requires prepared dataset shards; full training should run on a CUDA device with BF16 support.

## Coding Style & Naming Conventions

Use four-space indentation and conventional Python naming: `snake_case` for functions and variables, `PascalCase` for classes, and descriptive lowercase module names. Keep model math, training control, dataset I/O, and telemetry separated. Centralize hyperparameters and remote repository identifiers in `UltronConfig`; avoid hardcoded duplicates in scripts. Prefer established PyTorch, Accelerate, Transformers, and lm-eval APIs over custom infrastructure.

No formatter is currently enforced. Keep imports minimal, add type hints to public interfaces, and run `git diff --check` before submitting.

## Testing Guidelines

Tests use `pytest` and follow `test_<behavior>` naming. New model behavior should include small CPU fixtures. Cover tensor shapes, causal isolation, KV-cache equivalence, checkpoint compatibility, optimizer partitioning, and finite loss. Avoid tests that download data or instantiate the full model unless explicitly marked as slow.

## Commit & Pull Request Guidelines

History follows Conventional Commit prefixes such as `feat:`, `fix:`, `docs:`, and `refactor:`. Keep commits focused and imperative.

Pull requests should explain motivation, implementation, verification commands, and any checkpoint or dataset compatibility impact. Link relevant issues and include benchmark evidence for performance or model-quality claims. Add screenshots only for visual telemetry or documentation changes.

## Security & Configuration

Pass Hugging Face credentials through `HF_TOKEN`; never commit tokens. Treat optimizer checkpoints and pickle-based state as trusted local artifacts only.
