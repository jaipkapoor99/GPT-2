# General Software Engineering (SDE) Principles & Rules

## 1. Separation of Concerns (Single Responsibility Principle)

- Keep core model, training loop, and data pipeline logic cleanly separated from I/O, progress bar rendering, and external service logging.
- Move experiment tracking, W&B metrics, ETA calculations, and progress bar rendering into dedicated telemetry modules (e.g. `telemetry.py`).

## 2. Resource Lifecycle & Signal Hygiene

- Ensure streaming data loaders and background scripts catch `SIGINT`/`SIGTERM`, atomically persist recoverable state, and close iterators, file handles, and progress bars. Avoid abrupt `os._exit()` calls that bypass cleanup.

## 3. Dead Code Elimination & Schema Integrity

- Actively prune unused parameters, obsolete layer attributes, and legacy positional embedding artifacts (`wpe`) as model architectures evolve to maintain a clean codebase.

## 4. Deterministic Environment Execution

- Standardize Virtual Environment activation and dependencies (`uv`) in project configuration files (`.vscode/settings.json`) to prevent system Python environment drift.

## 5. Centralized Configuration & Single Source of Truth

- Centralize all project-wide constants, hyper-parameters, and remote target identifiers (`hf_repo_id`, `hf_dataset_repo_id`) within `config.py` (`UltronConfig`) so downstream scripts pull configuration dynamically without hardcoded duplicates.

## 6. Non-Blocking Terminal UI Logging

- Route all textual progress messages, checkpoint alerts, and status notifications through `pbar.write()` (e.g. `telemetry.print_message()`) rather than standard `print()` to prevent terminal cursor freezing and character corruption.

## 7. Programmatic Telemetry & Data Extraction

- Use W&B's native `_step` axis; do not create a user-visible step metric. Keep ETA and progress bookkeeping out of charts. Maintain continuous throughput and dev-loss series, an interval-average train/dev comparison, and explicit run-summary values.
- Prefer local `.wandb` history when it is sufficient; use the official `wandb.Api()` only when online state is required.

## 8. Reuse Established Libraries & Standard Frameworks (Don't Reinvent the Wheel)

- Always leverage battle-tested open-source libraries and official frameworks (e.g. EleutherAI `lm-evaluation-harness`, Hugging Face `datasets` / `transformers`, `tqdm`, `accelerate`) rather than writing custom ad-hoc implementations or reinventing existing tooling.

## 9. Dataset Geometry & Coverage

- Use a dataset stride equal to the configured context length. Smaller strides silently count overlapping source tokens multiple times.
- Shuffle training with deterministic epoch-specific permutations derived from `UltronConfig.data_seed`. Keep validation sequential and split at shard boundaries.
- Report both model-processed tokens and unique-corpus coverage when evaluating a sampling design.
- Keep DataLoader dataset state cheap to pickle under Python 3.14 `forkserver`. Serialize shard paths and metadata only; open and cache NumPy memmaps lazily inside each worker process.
- Run complete validation as its own timestamped W&B job. Log running loss, throughput, progress, final perplexity, elapsed time, and processed counts while retaining the atomic local JSON report.

## 10. Exact Resume Is a Cross-Cutting Contract

- Treat sampler epoch, batch offset, gradient accumulation, optimization state, W&B identity, shuffle seed, and validation cursor as one resume contract.
- Reject configurations that cannot reconstruct the exact data position. Never claim exact resume from model weights and optimizer state alone.

## 11. Validation Semantics

- Frequent validation is a rotating sampled estimate, not a final loss. Advance `dev_batch_cursor`, wrap at exhaustion, and persist it in checkpoints.
- Use `scripts/validate.py` for a complete leakage-safe validation pass.

## 12. Test Failure Modes

- Every behavioral fix requires a regression test. Cover malformed state, truncated artifacts, incompatible metadata, boundary arithmetic, resume transitions, and negative inputs—not only happy paths.
- Keep the default suite CPU-safe and network-free; gate compiler and CUDA checks explicitly.
