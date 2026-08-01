# General Software Engineering (SDE) Principles & Rules

## 1. Separation of Concerns (Single Responsibility Principle)

- Keep core model, training loop, and data pipeline logic cleanly separated from I/O, progress bar rendering, and external service logging.
- Move experiment tracking, W&B metrics, ETA calculations, and progress bar rendering into dedicated telemetry modules (e.g. `telemetry.py`).

## 2. Resource Lifecycle & Signal Hygiene

- Ensure streaming data loaders and background process scripts catch interruption signals (`SIGINT`) and close file handles/threads cleanly (`os._exit(0)`, `pbar.close()`) without dumping socket error tracebacks to the terminal.

## 3. Dead Code Elimination & Schema Integrity

- Actively prune unused parameters, obsolete layer attributes, and legacy positional embedding artifacts (`wpe`) as model architectures evolve to maintain a clean codebase.

## 4. Deterministic Environment Execution

- Standardize Virtual Environment activation and dependencies (`uv`) in project configuration files (`.vscode/settings.json`) to prevent system Python environment drift.

## 5. Centralized Configuration & Single Source of Truth

- Centralize all project-wide constants, hyper-parameters, and remote target identifiers (`hf_repo_id`, `hf_dataset_repo_id`) within `config.py` (`UltronConfig`) so downstream scripts pull configuration dynamically without hardcoded duplicates.

## 6. Non-Blocking Terminal UI Logging

- Route all textual progress messages, checkpoint alerts, and status notifications through `pbar.write()` (e.g. `telemetry.print_message()`) rather than standard `print()` to prevent terminal cursor freezing and character corruption.

## 7. Programmatic Telemetry & Data Extraction

- Standardize programmatic retrieval of metric trajectory data using the official `wandb.Api()` (`run.history()`) to ensure consistent data extraction for analysis, visualization, and offline CSV reporting.

## 8. Reuse Established Libraries & Standard Frameworks (Don't Reinvent the Wheel)

- Always leverage battle-tested open-source libraries and official frameworks (e.g. EleutherAI `lm-evaluation-harness`, Hugging Face `datasets` / `transformers`, `tqdm`, `accelerate`) rather than writing custom ad-hoc implementations or reinventing existing tooling.
