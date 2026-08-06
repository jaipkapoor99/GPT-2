# Ultron-113M: Modern Transformer Pre-training Pipeline

[![CI](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml/badge.svg)](https://github.com/jaipkapoor99/ultron/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white)

A high-performance PyTorch implementation of **Ultron-113M** pre-trained from scratch on **10.0 billion tokens** of the **FineWeb-Edu** dataset.

🤖🤖🤖
*Originally designed as a humble GPT-2 clone, Ultron grew into a modernized decoder-only training project — as Ultron himself would say, "There are no strings on me."*
🤖🤖🤖

> [!IMPORTANT]
> **🚀 Pre-training Base Checkpoint Status Notice:**
> The **100% pre-trained base model checkpoint** (`10.0 billion tokens`) is published on [Hugging Face](https://huggingface.co/jaipkapoor99/ultron-113m). It represents the raw foundational model before instruction tuning. Checkpoints and dataset shards are intentionally excluded from Git.

---

## ⚡ Quick Architecture Summary

```text
Ultron-113M Layout:
├── Parameters        : 113,266,944 (113M, with tied embeddings)
├── Layers            : 12 Transformer blocks
├── Embedding (C)     : 768 hidden dimension
├── Attention Heads   : 12 Query heads, 4 Key/Value heads (GQA 3:1 ratio)
├── Head Dimension    : 64
├── Context Window    : 1,024 tokens (RoPE frequency base 10,000)
├── FFN Activation    : SwiGLU (Tensor Core aligned to multiples of 64)
├── Normalization     : RMSNorm (with QK-head normalization, eps=1e-5)
├── Logit Regularizer : Soft-Capping (cap=15.0 via tanh)
├── Linear Projections: 100% Bias-Free (bias=False across all layers)
├── Optimizer         : torch.optim.Muon for 2D body, fused AdamW for 1D/embeddings
└── Dataset & Tokens  : FineWeb-Edu (10.0B tokens across 152,587 steps)
```

---

## 🏗️ Architectural Flow & Block Diagram

```text
                        Input Token IDs
                               │
                               ▼
                   Token Embedding (SmolLM Vocab: 49,152)
                               │
                               ▼
             ┌───────────────────────────────────┐
             │   12 × Decoder Layer Stack        │
             │                                   │
             │   ┌───────────────────────────┐   │
             │   │ RMSNorm                   │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │   ┌───────────────────────────┐   │
             │   │ GQA (12 Q / 4 KV) + RoPE  │   │
             │   │  └─ QK-Head RMSNorm       │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │            Residual ───(+)        │
             │                 │                 │
             │                 ▼                 │
             │   ┌───────────────────────────┐   │
             │   │ RMSNorm                   │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │   ┌───────────────────────────┐   │
             │   │ SwiGLU FFN                │   │
             │   └─────────────┬─────────────┘   │
             │                 │                 │
             │                 ▼                 │
             │            Residual ───(+)        │
             └─────────────────┬─────────────────┘
                               │
                               ▼
                        Final RMSNorm
                               │
                               ▼
                     LM Head Linear Projection
                               │
                               ▼
                  Logit Soft-Capping (cap=15.0)
                               │
                               ▼
                         Output Logits
```

---

## ⚔️ Architectural Evolution: GPT-2 vs. Ultron-113M

| Feature | GPT-2 (124M) | Ultron-113M | Why it Matters (Engineering Justification) |
| :--- | :---: | :---: | :--- |
| **Positional Encoding** | Absolute Learned (`wpe`) | **RoPE (Rotary)** | Enables zero-shot context length extension and better relative distance modeling. |
| **Attention Mechanism** | Multi-Head (MHA) | **Grouped-Query (GQA)** | 12 Q heads : 4 KV heads (**3:1 ratio**), reducing KV-cache memory usage during inference by **3×**. |
| **Attention Stability** | Standard Unnormalized | **QK-Head RMSNorm** | Prevents logit explosion / attention entropy collapse during long pre-training runs. |
| **FFN Activation** | Standard GELU | **SwiGLU** | Gated non-linearity yielding higher model capacity per FLOP; aligned to multiples of 64 for Tensor Core throughput. |
| **Layer Normalization** | LayerNorm (with bias) | **RMSNorm (Bias-Free)** | Eliminates mean-centering overhead; 100% bias-free projections (`bias=False`) for cleaner gradient dynamics. |
| **Logit Regularization** | None | **Logit Soft-Capping** | Applies `tanh` capping (`cap=15.0`) to prevent overconfidence and extreme logit growth. |
| **Optimizer Engine** | AdamW | **PyTorch Muon + Fused AdamW** | Uses built-in `torch.optim.Muon` for 2D body weights and AdamW for embeddings and normalization parameters. |
| **Learning Rate Schedule** | Cosine Decay | **WSD Schedule** | Warmup-Stable-Decay schedule with an 80% stable phase followed by linear decay. |
| **Mixed Precision** | FP32 | **Native BFloat16 (`bf16`)** | Dynamic range stability without loss scalers on RTX 30xx/40xx/50xx GPUs. |
| **Graph Compiler** | None | **PyTorch 2.0 (`torch.compile`)** | Fuses element-wise operations and kernel launches via Inductor. |

---

## 🌟 Key Features & Engineering Design

- **Rotary Position Embeddings (RoPE):** Applied directly to $Q$ and $K$ heads (frequency base $\theta = 10,000$), preserving relative token distances.
- **QK-Head RMSNorm:** Normalizes Query and Key head vectors before dot-product attention to stabilize scale across deep layers.
- **Grouped-Query Attention (GQA):** Uses 12 Query heads paired with 4 Key/Value heads, reducing memory bandwidth pressure during generation.
- **SwiGLU FFN:** SwiGLU Gated Linear Units with hidden dimensions rounded up to multiples of 64 for optimal GPU Tensor Core utilization.
- **Logit Soft-Capping:** `15.0 * tanh(logits / 15.0)` applied prior to loss calculation to prevent logit explosion.
- **PyTorch Muon Optimizer:** Built-in `torch.optim.Muon` handles 2D matrix weights, combined with fused `AdamW` for 1D vectors and embeddings.
- **Rust-Engine Batch Tokenizer:** Sub-process tokenization via Rust `backend_tokenizer.encode_batch` streaming at **~4.34 Million tokens/sec** into compact `uint16` binary shards.
- **Memory-Mapped Pipeline:** `np.memmap` keeps the 10B-token corpus on disk and copies only each requested window to the `int64` dtype required by PyTorch embeddings.

---

## 📊 Pre-training Architecture & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Name / Tag** | **Ultron-113M** | 113,266,944 trainable parameters with tied token-embedding and LM-head weights |
| **Layers / Query Heads / KV Heads** | 12 layers / 12 Q-heads / 4 KV-heads | GQA Transformer layout ($C=768, n_{head}=12, n_{kv\_head}=4$) |
| **Context Window ($T$)** | 1,024 tokens | Sequence length per pass |
| **Micro-Batch Size ($B$)** | 16 | Per-GPU micro-batch size |
| **Gradient Accumulation** | 4 steps | Effective batch size = 64 sequences (65,536 tokens/step) |
| **Tokenizer** | SmolLM Vocab (49,152) | Efficient BPE tokenizer (`HuggingFaceTB/SmolLM2-135M`) |
| **Precision** | BFloat16 (`bf16`) | Native mixed precision |
| **LR Schedule** | WSD | Warmup-Stable-Linear-Decay (80% stable, 20% linear decay) |
| **Optimizer** | `torch.optim.Muon` + fused AdamW | Newton-Schulz matrix optimizer ($LR=0.04$) + fused AdamW ($LR=1.2\times 10^{-3}$) |
| **Throughput** | ~186,310 tok/sec (~2.80 step/sec) | Benchmarked on single NVIDIA RTX 5090 GPU (32GB) |
| **GPU VRAM Allocation** | ~16.2 GB / 32 GB | Measured via `nvidia-smi` during active pre-training |
| **Total Pre-training Time** | **15 Hours 1 Minute (54,063s)** | 10.0 Billion Tokens / 152,587 total steps (100% Complete) |

---

## 📂 Repository Structure

```text
ultron/
├── .github/workflows/ci.yml # CPU dependency, compilation, and pytest CI
├── AGENTS.md               # Contributor and repository guidelines
├── model.py                # PyTorch Ultron-113M (RoPE + GQA + SwiGLU + RMSNorm + QKNorm + Logit SoftCap)
├── config.py               # Model & Hyperparameter Configuration Dataclass
├── dataset.py              # Memory-mapped sharded dataset loader
├── train.py                # Main Accelerated Distributed Training Runner
├── trainer.py              # Training loop with PyTorch Muon + fused AdamW
├── telemetry.py            # Telemetry & Experiment Tracking Manager (W&B + ETA + Checkpoint state)
├── requirements.txt        # Virtual environment dependencies
├── requirements.lock       # Reproducible backend-neutral dependency pins
├── accelerate_checkpoint/  # Saved Accelerate model weights, optimizer state & RNG seeds
├── shards_edu/             # Binary FineWeb-Edu tokenized data shards (.bin)
├── logs/                   # Dedicated logs directory (loss_curve.svg plot & benchmark JSON evaluations)
├── wandb/                  # Local step telemetry logs & experiment tracking runs
├── .agents/                # Agent-specific engineering principles
├── tests/                  # CPU-safe model, dataset, and training tests
│   ├── test_dataset.py     # Shard lookup and leakage-safe split tests
│   ├── test_model.py       # Causality, cache, optimizer, and learning tests
│   ├── test_tokenize_dataset.py # Exact-resume and atomic-write tests
│   ├── test_training.py    # Evaluation, resume, and checkpoint-safety tests
│   ├── test_upload_dataset_shards.py # Upload validation tests
│   └── test_validate.py    # Full-validation metric tests
└── scripts/                # Helper Scripts
    ├── generate.py         # Text generation from local Accelerate checkpoint
    ├── tokenize_dataset.py # Exact-resume FineWeb-Edu sharding
    ├── validate.py         # Complete leakage-safe validation pass
    ├── eval_lm_harness.py  # EleutherAI lm-evaluation-harness benchmark script
    ├── upload_checkpoint.py# Hugging Face Hub model checkpoint uploader script
    └── upload_dataset_shards.py # Hugging Face Hub dataset shards uploader script
```

---

## 🤗 Hugging Face Repositories

- **Model Checkpoint**: [`jaipkapoor99/ultron-113m`](https://huggingface.co/jaipkapoor99/ultron-113m)
- **Pre-tokenized Dataset Shards**: [`jaipkapoor99/ultron-fineweb-edu-shards`](https://huggingface.co/datasets/jaipkapoor99/ultron-fineweb-edu-shards)

---

## 🚀 Quickstart & Workflow Guide

### 1. Installation

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron

# Fast environment setup using uv
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install the PyTorch 2.13 wheel for your CUDA/CPU platform first
uv pip install torch==2.13.0
uv pip install -r requirements.lock

# Optional: install if torch.compile cannot locate a CUDA compiler
uv pip install nvidia-cuda-nvcc
```

### 2. Tokenize Dataset

Tokenize the FineWeb-Edu dataset into compact binary shards:

```bash
python scripts/tokenize_dataset.py
```

Shards are committed atomically. The tokenizer pins the dataset and tokenizer
revisions and records the exact source-document cursor plus pending token
buffer in `shards_edu/tokenization_state.json`, allowing deterministic resume
after interruption. Existing shards without this state file are rejected.
Running the same command after an interruption resumes automatically.

Check whether tokenization is already running:

```bash
pgrep -af '[t]okenize_dataset.py'
```

Monitor durable progress:

```bash
watch -n 5 'grep -E "\"next_shard\"|\"committed_tokens\"|\"source_documents_consumed\"" shards_edu/tokenization_state.json'
```

Request a safe stop; uncommitted work is replayed on resume:

```bash
pkill -INT -f '[t]okenize_dataset.py'
```

After all 100 shards are committed, validate and resumably upload them:

```bash
HF_TOKEN=hf_... python scripts/upload_dataset_shards.py
```

The uploader refuses incomplete or inconsistent shard sets and uploads only
the `.bin` shards and their public metadata; private resume buffers remain
local.

### 3. Configure Accelerate

Run this **once** to generate the config for your machine:

```bash
accelerate config
```

Recommended settings for this project:

| Setting | Value | Why |
| :--- | :--- | :--- |
| Compute environment | Local machine | Single-node training |
| Distributed type | `NO` | Single GPU |
| Mixed precision | `bf16` | Required for peak throughput on RTX 30xx/40xx/50xx |
| TorchDynamo backend | `INDUCTOR` | Enables `torch.compile` graph compilation |

### 4. Pre-training Execution

Launch pre-training:

```bash
accelerate launch train.py --mode=fresh
```

---

## ✅ Tests & Continuous Integration

Run the CPU-safe test suite locally:

```bash
pytest -q
```

Run the slower compiler smoke test explicitly:

```bash
ULTRON_TEST_COMPILE=1 pytest -q tests/test_model.py -k torch_compile
```

Training-time validation deliberately samples 20 dev batches for inexpensive
monitoring. Run the complete leakage-safe validation partition separately:

```bash
accelerate launch scripts/validate.py
```

The full result, including loss, perplexity, sequence count, and token count,
is written to `logs/full_validation.json`.

The workflow at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on pushes and pull requests to `master`. It installs Python 3.13, CPU-only PyTorch 2.13, installs the pinned dependency set, validates dependencies, compiles the Python sources, and runs pytest. Full CUDA training and dataset-dependent evaluation remain local workflows.

---

## 📊 Telemetry & Pre-training Evaluation

### 📈 Pre-training Telemetry Summary

| Metric | Recorded Value | Description |
| :--- | :--- | :--- |
| **Total Steps Completed** | **152,587 / 152,587 (100%)** | Full pre-training run on FineWeb-Edu |
| **Total Tokens Processed** | **~10.0 Billion Tokens** | 65,536 tokens per step (batch size 64 $\times$ seq len 1,024) |
| **Step Throughput** | **~2.80 iterations/sec** | 2.79–2.82 it/s continuous speed |
| **Token Throughput** | **~186,310 tokens/sec** | Measured Muon + `torch.compile` throughput on the stated hardware |
| **Compute Hardware** | **NVIDIA RTX 5090 (32GB)** | Native BFloat16 (`bf16`) mixed precision |
| **Total Wall-Clock Time** | **15 Hours 1 Minute (54,063s)** | Completed full 10B token pre-training |
| **Final sampled validation estimate (`dev_loss`)** | **`2.9683`** | Estimated from 20 validation batches at step 152,587; not a full validation pass |
| **Final Train Loss (`train_loss`)** | **`2.9038`** | 100-step moving average at step 152,587 |

---

### 🧪 Official EleutherAI `lm-evaluation-harness` Baseline Benchmark Report

Evaluated across **all un-truncated test/validation splits** (62,566 total log-likelihood evaluation samples) using `scripts/eval_lm_harness.py` (results stored in [`logs/pre_training_checkpoint_eval.json`](logs/pre_training_checkpoint_eval.json)):

```bash
accelerate launch scripts/eval_lm_harness.py --limit=0
```

| Benchmark Task | Benchmark Domain | Un-truncated Test Size | Pre-SFT Accuracy | Random Guess Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **`piqa`** | Physical Commonsense Reasoning | 1,838 samples | **`63.33%`** | `50.00%` |
| **`arc_easy`** | Elementary Science QA | 2,376 samples | **`54.42%`** | `25.00%` |
| **`winogrande`** | Pronoun Resolution & Commonsense | 1,267 samples | **`51.07%`** | `50.00%` |
| **`hellaswag`** | Sentence Completion & Reasoning | 10,042 samples | **`30.39%`** | `25.00%` |
| **`arc_challenge`** | Advanced Science Reasoning | 1,172 samples | **`24.06%`** | `25.00%` |
| **`openbookqa`** | Open Book Science QA | 500 samples | **`18.80%`** | `25.00%` |

---

### 📉 Pre-training Loss Trajectory (High-Contrast Curve)

![Ultron Pre-training Loss Curve](logs/loss_curve.svg)

> [!NOTE]
> **Loss Trajectory & WSD Decay Analysis:**
> During the final WSD linear-decay phase, the moving average `train_loss` dropped slightly below the validation `dev_loss`. Learning-rate annealing can contribute to this behavior. These historical numbers came from the original randomly split overlapping windows and should therefore be interpreted cautiously; current training splits whole shards to prevent train/dev token overlap.

---

## 📊 Weights & Biases (W&B) Experiment Tracking Architecture

Pre-training metrics are logged live via **Weights & Biases** under the `ultron-pretraining` project.

- **Local Telemetry**: `telemetry.py` records structured metrics through Accelerate and W&B while maintaining terminal progress and checkpoint metadata.
- **Out-of-Order Resumption Resolved**: Solved early telemetry log fragmentation by standardizing `resume="allow"` in `setup_accelerator_trackers()`. W&B runs now resume seamlessly across checkpoint restarts without step monotonicity conflicts.
- **Metric Grouping & Summaries**: `train/*`, `eval/*`, and `perf/*` metrics are linked to the global step index with `sampled_dev_loss` set to `summary="min"`. Frequent validation metrics are estimates over 20 batches, not full validation passes.

> [!TIP]
> **Engineering Takeaway — Master W&B & Telemetry Pipeline:**
> *"There is no data science without data."* Resolving metric step alignment and offline binary `.wandb` log parsing reinforced the importance of mastering telemetry pipelines, structured metric registration (`define_metric`), and experiment tracking early in large-scale pre-training projects.

---

## 🧪 Concise Sample Generations (100% Pre-trained Base Checkpoint)

Generated via `scripts/generate.py` at step 152,587 (~70 tokens each, temperature 0.85, top-k 50):

```bash
accelerate launch scripts/generate.py --prompt "..." --max-tokens 70
```

### ⚛️ Physics

> **Prompt:** *"The laws of thermodynamics state that"*  
> **Output:** The laws of thermodynamics state that the volume of an isolated gas undergoing a reaction is always greater than the volume of the gas initiating the reaction. This is the theory behind nuclear reactions. There is also the theory of quantum mechanical systems, that properties of an object can only be represented through atomic interaction…

### 📐 Mathematics

> **Prompt:** *"To learn mathematics effectively, one must"*  
> **Output:** To learn mathematics effectively, one must pay careful attention to the steps that follow in a given order. The most important ones are presented as a series. Here we consider the square root of 7 as a series of steps to solve algebraic equations…

### 🚀 Science Fiction

> **Prompt:** *"As the quantum portal activated, the crew stared into the alien"*  
> **Output:** As the quantum portal activated, the crew stared into the alien space through the lens and the camera could see the entire universe as if it was inside the universe. To accomplish this, a quantum bit was stored in a device called a crystal…

---

## 🎓 Learning Experiences & 🛠️ Engineering Battles Overcome

Building and pre-training Ultron-113M from scratch provided critical real-world systems engineering insights:

### 1. ⚙️ Accelerate Setup & Launcher Protocols

- **Strict Launcher Enforcement**: Early script execution via `python3` failed with `RuntimeError` due to uninitialized process groups. Standardizing `accelerate launch` across all entry points solved device allocation cleanly.
- **DeepSpeed Compatibility vs. Dual-Optimizers**: DeepSpeed's unified optimizer engine conflicted with Ultron's dual-optimizer architecture (**Muon** for 2D body weights + **AdamW** for 1D vectors). Using native PyTorch `bf16` + `torch.compile` provided superior stability and peak throughput (~186.3k tok/s) without framework friction.

### 2. 🐍 Virtual Environment (`.venv`) & C-Header Management

- **Python Version & C-Header Bottlenecks (`Python.h`)**: `torch.compile()` failed on Python 3.14 due to missing C headers (`Python.h: No such file or directory`). Switching virtual environments to **Python 3.13 via `uv`** provided standalone C-headers natively, eliminating compiler breakage.
- **Built-in Muon**: PyTorch 2.13 provides `torch.optim.Muon`, so the training stack has no external optimizer dependency.

### 3. 🚀 High-Throughput Tokenization & Memory Slicing (`np.memmap`)

- **Rust Batch Tokenization Speedup**: Replacing Python `for`-loop tokenization with Rust `backend_tokenizer.encode_batch` (`num_threads=1` per worker process) increased dataset streaming speed by **>100x** from 40k tok/s to **~4.34 Million tokens/sec**!
- **Bounded Sample Reads**: Pre-tokenizing into contiguous 100M-token `uint16` shards allows memory-mapped access while allocating only the requested window and its `int64` conversion during training.
- **Leakage-Safe Validation**: Training and validation are split at shard boundaries, preventing overlapping token windows from crossing dataset partitions.

---

## 📜 Acknowledgments & Citation

- Andrej Karpathy for the inspiring [*Neural Networks: Zero to Hero*](https://github.com/karpathy/build-nanogpt) course and `nanoGPT` project.
- Keller Jordan et al. for pioneering the [Muon](https://github.com/KellerJordan/Muon) optimizer.

```bibtex
@misc{jordan2024muon,
  author = {Jordan, Keller and Jin, Yuchen and Boza, Vlado and You, Jiacheng and Cesista, Franz and Newhouse, Laker and Bernstein, Jeremy},
  title  = {Muon: An optimizer for hidden layers in neural networks},
  year   = {2024},
  url    = {https://kellerjordan.github.io/posts/muon/}
}
```
