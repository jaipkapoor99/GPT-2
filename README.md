# Ultron (124M 2026 SOTA) Pre-training Pipeline

A high-performance, modern PyTorch implementation of **Ultron (124M parameters)** pre-trained from scratch on the **FineWeb-Edu** dataset, incorporating 2026 State-of-the-Art (SOTA) LLM training innovations.

🤖🤖🤖
*Originally designed as a humble GPT-2 clone, Ultron rapidly outgrew its original scope to become a 2026 SOTA powerhouse — as Ultron himself would say, "There are no strings on me."*
🤖🤖🤖

Features **Rotary Position Embeddings (RoPE)**, **QK-Head RMSNorm**, **Muon Newton-Schulz Matrix Optimizer**, **Grouped-Query Attention (GQA)**, **SwiGLU FFN Activations**, **Logit Soft-Capping**, **100% Bias-Free Linear Layers**, **WSD (Warmup-Stable-Decay) Schedule**, **Zero-Copy Memory-Mapped Data Pipeline**, and **PyTorch 2.0 Graph Compilation**.

---

## 🌟 Key Features & Architecture

- **Rotary Position Embeddings (RoPE):** RoPE (LLaMA 3 / Qwen 2.5 standard) applied to $Q$ and $K$ heads, enabling zero-shot context window extension.
- **QK-Head RMSNorm:** Query/key RMSNorm (Qwen 2.5 / Gemma 2 standard) for loss stability during pre-training.
- **Muon Newton-Schulz Matrix Optimizer:** Pre-training optimization using Keller Jordan's **Muon** (Momentum Orthogonalized by 5th-order Newton-Schulz iterations) for 2D body weights, paired with fused `AdamW` for 1D vectors/embeddings.
- **Grouped-Query Attention (GQA):** 12 Query heads and 4 Key/Value heads (3:1 Query-to-KV ratio), reducing KV-cache memory usage during autoregressive generation by **3x**.
- **SwiGLU FFN:** SwiGLU Gated Linear Units aligned to multiples of 64 for optimal GPU Tensor Core throughput.
- **Logit Soft-Capping:** Gemma 2 standard logit soft-capping (`cap=15.0`) applied via `tanh` to prevent overconfidence.
- **RMSNorm & Bias-Free Layers:** Root Mean Square Layer Normalization (RMSNorm) and bias-free linear projections across all layers (`bias=False`).
- **Warmup-Stable-Decay (WSD) Schedule:** WSD learning rate schedule (80% stable phase, 20% cosine decay).
- **Zero-Copy Data Loader:** Memory-mapped disk slicing (`np.memmap`) for zero RAM allocation overhead during multi-billion token streaming.

---

## 📊 Pre-training Architecture & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Parameters** | 114,053,376 (114M) | GQA + Bias-Free SOTA 124M scale |
| **Layers / Query Heads / KV Heads** | 12 layers / 12 Q-heads / 4 KV-heads | GQA Transformer layout ($C=768, n_{head}=12, n_{kv\_head}=4$) |
| **Positional Embedding** | RoPE (Rotary) | Dynamic frequency base ($10,000$); learned absolute `wpe` embeddings present but superseded by RoPE |
| **QK Normalization** | RMSNorm | Query/Key head normalization |
| **Logit Soft-Cap** | 15.0 | Gemma 2 style `tanh` soft-capping |
| **FFN Activation** | SwiGLU | Multiples of 64 Tensor-Core aligned |
| **Normalization** | RMSNorm | $\epsilon=10^{-5}$ |
| **Context Window ($T$)** | 1,024 tokens | Extendable sequence length |
| **Micro-Batch Size ($B$)** | 16 | Per-GPU micro-batch size |
| **Gradient Accumulation** | 4 steps | Effective batch size = 64 sequences (65,536 tokens/step) |
| **Tokenizer** | SmolLM Vocab (49,152) | Efficient BPE tokenizer |
| **Precision** | BFloat16 (`bf16`) | Native mixed precision |
| **LR Schedule** | WSD | Warmup-Stable-Linear-Decay (80% stable, 20% linear decay) |
| **Optimizer** | Muon + AdamW | Newton-Schulz matrix optimizer ($LR=0.04$) + fused AdamW ($LR=1.2\times 10^{-3}$) |

---

## 📂 Repository Structure

```text
ultron/
├── model.py                # PyTorch Ultron (RoPE + GQA + SwiGLU + RMSNorm + QKNorm + Logit SoftCap)
├── config.py               # Model & Hyperparameter Configuration Dataclass
├── dataset.py              # Zero-Copy Memmap Sharded Dataset Loader
├── train.py                # Main Distributed Accelerated Training Script
├── trainer.py              # Trainer Class with Keller Jordan Muon + AdamW
├── generate.py             # Text generation from local Accelerate checkpoint
├── requirements.txt        # Dependencies
├── loss_curve.svg          # Vector SVG pre-training loss trajectory plot
├── accelerate_checkpoint/  # Saved Accelerate model weights, optimizer state & RNG seeds
├── shards_edu/             # Binary FineWeb-Edu tokenized data shards (.bin)
├── wandb/                  # Local step telemetry logs & experiment tracking runs
├── .agents/                # Project AGENTS.md rules & workspace customization
├── tests/                  # Unit & Integration Tests (Accelerate + torch.testing)
│   └── test_model.py       # Core model architecture & generation unit tests
└── scripts/                # Helper Scripts
    └── tokenize_dataset.py # FineWeb-Edu dataset tokenization into binary shards
```

---

## 🚀 Quickstart & Workflow Guide

### 1. Installation

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron

# Fast environment setup using uv
uv venv --python 3.13 venv
source venv/bin/activate
uv pip install -r requirements.txt nvidia-cuda-nvcc
```

### 2. Tokenize Dataset

Tokenize the FineWeb-Edu dataset into compact binary shards:

```bash
python3 scripts/tokenize_dataset.py
```

### 3. Configure Accelerate ⚠️ Required

> [!IMPORTANT]
> **`accelerate config` is a cornerstone of this repository.** Every script — `train.py`, `generate.py`, and `tests/test_model.py` — is launched exclusively via `accelerate launch` and will raise a `RuntimeError` if invoked with plain `python3`. The config file (`~/.cache/huggingface/accelerate/default_config.yaml`) is the single source of truth for device, precision, and compiler settings.

Run this **once** to generate the config for your machine:

```bash
accelerate config
```

**Recommended settings for this project:**

| Setting | Value | Why |
| :--- | :--- | :--- |
| Compute environment | Local machine | Single-node training |
| Distributed type | No | Single GPU |
| Mixed precision | `bf16` | Required for SOTA throughput on RTX 30xx/40xx/50xx |
| TorchDynamo backend | `inductor` | Enables `torch.compile` graph compilation |

Once configured, **all three entry points** use this config automatically:

```bash
accelerate launch train.py           # pre-training
accelerate launch generate.py        # text generation
accelerate launch -m unittest tests.test_model  # unit tests
```

### 4. Pre-training Run

Start pre-training (defaults to `--mode=continue` to seamlessly resume existing checkpoints):

```bash
accelerate launch train.py
```

To explicitly specify training mode (`--mode=continue`, `--mode=fresh`, or `--mode=test`):

```bash
# Resume from existing checkpoint
accelerate launch train.py --mode=continue

# Start a fresh run from step 0
accelerate launch train.py --mode=fresh

# Run for a specific step count (e.g. 500 steps)
accelerate launch train.py --mode=continue --max-steps=500

# Run a quick 100-step test mode
accelerate launch train.py --mode=test
```

#### ⚡ Performance & Throughput Benchmarks

- **Step Throughput**: `~2.80 iterations/sec` (2.79–2.82 it/s)
- **Token Throughput**: `~183,000+ tokens/sec` (65,536 tokens / step)
- **Full Pre-training Time (10B Tokens / 152.5k steps)**: **~15 hours total** on an RTX 5090 GPU!

- **Percentage** — rounded integer progress over the full run.
- **ETA** — time remaining (`h m s`), computed from the throughput of the **current session only**, so it stays accurate after a resumed checkpoint.
- **Step counter** — `current / total` steps.

#### 💾 Checkpointing

Training state is saved as an **Accelerate checkpoint** (`accelerate_checkpoint/`) which includes model weights, optimizer state, scheduler, and RNG seeds. Resume is seamless:

```bash
accelerate launch train.py --mode=continue
```

> [!NOTE]
> Test runs (`--mode=test`) never write checkpoints, keeping production resume states unpolluted.

#### 📈 Experiment Tracking & Pre-training Telemetry Summary

Pre-training metrics are logged via **Weights & Biases** under the `ultron-pretraining` project.

##### 📊 Pre-training Run Telemetry & Performance Statistics

| Metric | Recorded Value | Description |
| :--- | :--- | :--- |
| **Total Steps Completed** | **152,587 / 152,587 (100%)** | Full pre-training run on FineWeb-Edu |
| **Total Tokens Processed** | **~10.0 Billion Tokens** | 65,536 tokens per step (batch size 64 $\times$ seq len 1,024) |
| **Step Throughput** | **~2.80 iterations/sec** | 2.79–2.82 it/s continuous speed |
| **Token Throughput** | **~183,400 tokens/sec** | SOTA Muon + PyTorch 2.0 compile throughput |
| **Compute Hardware** | **NVIDIA RTX 5090 (32GB)** | Native BFloat16 (`bf16`) mixed precision |
| **Total Wall-Clock Time** | **~15.0 Hours Total** | Completed full 10B token pre-training |
| **Final Validation (`dev_loss`)** | **2.9179** | Evaluated on validation set at step 152,587 |
| **Final Train Loss (`train_loss`)** | **~2.85** | 100-step moving average at step 152,587 |
| **Exported CSV Log Files** | **`wandb_export_*.csv`** | Step-by-step telemetry logs for steps 143,251–152,587 |

##### 📉 Sampled Loss Progression Checkpoints (WSD Cosine Decay Phase)

| Training Step | Validation Loss (`dev_loss`) | Step Train Loss (`train_loss`) |
| :--- | :--- | :--- |
| **Step 143,500** | `2.9476` | `2.9528` |
| **Step 145,500** | `2.9415` | `3.0203` |
| **Step 147,500** | `2.9354` | `3.0872` |
| **Step 149,500** | `2.9279` | `2.8706` |
| **Step 151,500** | `2.9217` | `3.1029` |
| **Step 152,500** | `2.9179` | `2.8702` |
| **Step 152,587 (100% Final)** | **`2.9179`** | **`2.8530`** |

##### 📉 Pre-training Loss Trajectory (High-Contrast Detailed Curve)

![Ultron Pre-training Loss Curve](loss_curve.svg)

> [!IMPORTANT]
> **Loss Trajectory & Overfitting Analysis:**
> Towards the final WSD cosine decay phase (steps 150,000–152,587), the moving average `train_loss` (~2.85) dropped slightly below the validation `dev_loss` (2.9179). This slight divergence indicates the onset of mild capacity saturation / slight overfitting on the pre-training dataset. If pre-training had been extended beyond 152,587 steps without tuning regularization hyperparameters (e.g., increasing weight decay or introducing dropout/data filtering), validation performance (`dev_loss`) would have begun to plateau and eventually degrade.
> [!WARNING]
> **Telemetry Log Fragmentation Notice:**
> Due to repeated local crashes, hardware reboots, and multiple run renamings during early hyperparameter exploration, portions of the early step-level Weights & Biases telemetry logs for `ultron-pretraining` were lost or fragmented across local run directories (`wandb/`). The model weights, final telemetry logs (`wandb_export_*.csv`), pre-training step count (152,587 / 152,587), and final state remain 100% intact and verified.

- `dev_loss` and `train_loss` are logged at the **same step** at eval intervals so they appear on a **single shared chart** in W&B.
- `dev_loss` is registered as a **summary metric (min)**, so the best validation loss is always visible on the run card.

### 5. Running Unit Tests

Run the test suite powered by `Accelerate` and `torch.testing`:

```bash
accelerate launch -m unittest tests.test_model
```

## 🧪 Sample Generations (Step 152,587 / 152,587 — 100% Pre-trained)

These samples were generated by `generate.py` across **6 different domains** upon 100% pre-training completion (temperature 0.85, top-k 50, 150 new tokens each).

```bash
accelerate launch generate.py --prompt "..." --temperature 0.85 --top-k 50 --max-tokens 150
```

---

### ⚛️ Physics

**Prompt:** *"The laws of thermodynamics state that"*

> The laws of thermodynamics state that the volume of an isolated gas undergoing a reaction is always greater than the volume of the gas initiating the reaction. This is the theory behind the theory of nuclear reactions. There is also the theory of quantum mechanical systems, that the properties of an object can only be represented through an interaction of those atoms in a system. This is known as the Schrödinger equation. This theory has been proven by the discovery of Quantum electrodynamics, which holds that the energy in the field of an atom is a function of its charge. This can be solved by substituting energy into the formula for the energy in the atom…

---

### 📐 Education & Mathematics

**Prompt:** *"To learn mathematics effectively, one must"*

> To learn mathematics effectively, one must pay careful attention to the steps that follow (in a given order). The most important ones are: In the following example, the method of the square root of 7 is presented as a series. Here we consider the square root of 7 as a series of steps. Example of square root of 7. In the following example, we will consider the following method of solving the square root of 7…

---

### 🏛️ History

**Prompt:** *"The ancient Romans built their empire by"*

> The ancient Romans built their empire by the aid of ships which they used to travel in shallow waters. The Roman Empire was originally comprised of four areas: the North (Italy), the South (North-West Europe) and the East (northern France). It reached its peak around the middle of the 4th century AD, and lasted until the beginning of the 1st century AD. The Romans controlled and controlled the Mediterranean Sea, which is home to the world’s largest known ocean, the Red Sea. It was the center of global trade and commerce for many centuries…

---

### 🌍 Environment & Climate

**Prompt:** *"Climate change is affecting our planet by"*

> Climate change is affecting our planet by increasing the frequency and intensity of extreme weather events, including severe droughts, floods and heat waves. More research is needed to understand why these climate effects are happening so suddenly, in ways that are difficult to model and could have a powerful impact on weather and climate. The study includes a series of climate models that have been developed under the auspices of the U.S. National Oceanic and Atmospheric Administration and the University of Illinois at Urbana-Champaign…

---

### ✍️ Creative Writing

**Prompt:** *"She opened the old wooden door and"*

> She opened the old wooden door and said it was a bad idea to go to the hospital. He also added that the hospital would not allow her to have surgery. The doctors said that the patient had never seen a surgeon before, and that he had been to the hospital before and the doctor had not told him about his surgery. She said that he would not help him, but asked the doctor to say that his life was in danger. After the doctor gave him the news, they began to talk about their plan…

---

### 🧠 Biology & Neuroscience

**Prompt:** *"The human brain is capable of"*

> The human brain is capable of handling some of the most complex stimuli, as well as the most complex responses to those stimuli. The brain can be thought of as a computer system. If you are not part of the software team, it is our job to be an algorithm and a machine. You may or may not have a brain and it is your role to teach that skill to someone else. You may or may not have the skills to program that machine…

---

> [!NOTE]
> **Benchmarking & Pre-training Completion Notice:**
>
> 1. **Completion Snapshot**: The text generation samples above represent the final **100% pre-training completion snapshot** at step **152,587** (~10B tokens).
> 2. **Future Evaluation Plan**: Formal zero-shot benchmark evaluation across standard benchmark suites (`HellaSwag`, `MMLU`, `GSM8K`, `HumanEval`, etc.) is deferred to the upcoming Supervised Fine-Tuning (SFT) / instruction-tuning phase, as evaluating raw pre-trained base models directly on task prompts does not accurately reflect model capability prior to instruction alignment.

---

## 🎓 Learning Experiences

Building and pre-training Ultron from scratch provided key technical and operational insights:

- **Transitioning from WSL to Native Ubuntu 24.04 LTS**: Shifting from Windows Subsystem for Linux (WSL) to native Ubuntu 24.04 LTS eliminated GPU driver virtualization overhead, resolved CUDA memory allocation bottlenecks, and drastically improved PyTorch `torch.compile` Triton graph compilation stability on RTX 50-series hardware.
- **Experiment Tracking with Weights & Biases (W&B)**: Learned foundational W&B step logging and metric tracking, though full programmatic API automation remains a work in progress (requiring manual CSV log exports for final trajectory plotting rather than automated API data fetching).

---

## 📜 Acknowledgments

- Andrej Karpathy for the inspiring [*Neural Networks: Zero to Hero*](https://github.com/karpathy/build-nanogpt) course and `nanoGPT` project.
- Keller Jordan et al. for pioneering the [Muon](https://github.com/KellerJordan/Muon) optimizer and algorithmic speedrun innovations.

---

## 📚 Citation

If you use the Muon optimizer or this codebase, please cite the original Muon work:

```bibtex
@misc{jordan2024muon,
  author = {Jordan, Keller and Jin, Yuchen and Boza, Vlado and You, Jiacheng and Cesista, Franz and Newhouse, Laker and Bernstein, Jeremy},
  title  = {Muon: An optimizer for hidden layers in neural networks},
  year   = {2024},
  url    = {https://kellerjordan.github.io/posts/muon/}
}
```
