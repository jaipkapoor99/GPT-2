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

> [!NOTE]
> **Hugging Face Pipeline (Parked / Under Development):** Hugging Face Hub integration, `pipeline("text-generation")` wrappers (`UltronForCausalLM`), and HF export scripts (`export_to_hf.py`, `generate_from_hf.py`) are parked and under active development. Core development focus is centered on native local pre-training, benchmarking.

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
├── model.py              # PyTorch Ultron (RoPE + GQA + SwiGLU + RMSNorm + QKNorm + Logit SoftCap)
├── config.py             # Model & Hyperparameter Configuration Dataclass
├── hf_model.py           # Hugging Face Wrapper (UltronHFConfig + UltronForCausalLM)
├── model_card.yaml       # HF Model Card Metadata Specification
├── dataset.py            # Zero-Copy Memmap Sharded Dataset Loader
├── train.py              # Main Distributed Accelerated Training Script
├── trainer.py            # Trainer Class with Keller Jordan Muon + AdamW
├── generate.py           # Text generation from local Accelerate checkpoint
├── requirements.txt      # Dependencies
├── README.md             # Project Overview & Architecture Guide
├── tests/                # Unit & Integration Tests (Accelerate + torch.testing)
│   ├── test_model.py
│   └── test_hf_wrapper.py
└── scripts/              # Helper Scripts
    ├── tokenize_dataset.py # FineWeb-Edu dataset tokenization into binary shards
    └── export_to_hf.py     # Local export & Hugging Face Hub upload pipeline
```

---

## 🚀 Quickstart & Workflow Guide

### 1. Installation

```bash
git clone https://github.com/jaipkapoor99/ultron.git
cd ultron
pip install -r requirements.txt
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

#### 🖥️ Training Progress Display

During training a single-line progress counter is printed to the terminal and continuously overwritten (no clutter):

```text
[12%] ETA: 2h 45m 30s | Step: 18312/152587
```

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

#### 📈 Experiment Tracking (W&B)

All runs are tracked with **Weights & Biases** under the `ultron-pretraining` project. The following metrics are logged at every step:

| Metric | Description |
| :--- | :--- |
| `train_loss` | Training cross-entropy loss (every step) |
| `dev_loss` | Validation cross-entropy loss (every `eval_interval` steps, co-plotted with `train_loss`) |
| `lr` | Current learning rate |
| `eta_seconds` | Estimated time remaining (seconds), session-scoped |
| `progress_percent` | Integer percentage of total run |

- `dev_loss` and `train_loss` are logged at the **same step** at eval intervals so they appear on a **single shared chart** in W&B.
- `dev_loss` is registered as a **summary metric (min)**, so the best validation loss is always visible on the run card.

### 5. Running Unit Tests

Run the test suite powered by `Accelerate` and `torch.testing`:

```bash
accelerate launch -m unittest tests.test_model
```

## 🧪 Sample Generations (Step 51,750 / 152,587 — ~34% trained)

These samples were generated by `generate.py` across **6 different domains** at ~34% training progress (temperature 0.85, top-k 50, 150 new tokens each).

```bash
accelerate launch generate.py --prompt "..." --temperature 0.85 --top-k 50 --max-tokens 150
```

---

### ⚛️ Physics

**Prompt:** *"The laws of thermodynamics state that"*

> The laws of thermodynamics state that energy is a product of movement of a particle. When this particle is moving at constant speed, energy is conserved by the motion of the particle. This is the law of conservation of energy. The energy of a particle, or its energy, is equal to the mass of the particle, or its charge, and can be expressed in terms of the joule, or the second, or the joule of energy. The energy of a particle can be expressed as part of its angular momentum. To find energy, we need to find the angular momentum. Energy is equal to Joule of energy, or Joule mass. This type of energy is often called the energy of motion. It is not used as the basis for calculating energy in…

---

### 📐 Education & Mathematics

**Prompt:** *"To learn mathematics effectively, one must"*

> To learn mathematics effectively, one must understand a variety of techniques in order to become fluent and proficient in the subject. One of the best techniques in mathematics is the formula. Understanding the formula, which is essential in the field of mathematics, is crucial for comprehending certain concepts and achieving desired results. The formula helps comprehend the relationship between a given statement, the set of all coefficients, and the product of two variables. This formula is crucial for understanding the formulas of linear, quadratic, and exponential functions. It is also vital in mastering the theory of linear equations and the relationships between two variables. In addition, the formula helps grasp the fundamental concepts and forms of the equations. It helps grasp essential concepts such as coefficients and derivative…

---

### 🏛️ History

**Prompt:** *"The ancient Romans built their empire by"*

> The ancient Romans built their empire by laying the foundations of a thousand cities, by building walls as well, so if in 1000 it is not, we can imagine that its foundations were so advanced that it gave the Romans the largest empire. The Romans built their empire on the foundations of pyramids and temples. The Romans made pyramids out of stones, the Romans invented a machine to make them from pieces of stone. Their inventions are not really true, because the Romans are still very superstitious about pyramids. The Romans built the pyramids on the principles that the ancient rulers taught and that their rule was based on the foundations of the pyramids. The Romans, who were very different from the ancient rulers, were the first to build pyramids…

---

### 🌍 Environment & Climate

**Prompt:** *"Climate change is affecting our planet by"*

> Climate change is affecting our planet by increasing temperatures, and increasing precipitation. The impact on flooding is a result of the fact that climate change can be a major threat to human activity and the impact of flooding events on our environment. Climate change and flooding are not the same thing. Climate change impacts are linked to different types of precipitation, but the effect on flooding is much more complex. A large proportion of flooding in the UK is caused by climate change: there are three main types of climate change which account for less than 10% of the UK's average annual rainfall. Climate change is the main climate change driver in the UK, with the majority of UK precipitation having been caused by climate change. A lot of people, especially those with low rainfall…

---

### ✍️ Creative Writing

**Prompt:** *"She opened the old wooden door and"*

> She opened the old wooden door and sawed out one half of it into the house — the door was now open. It was a small church, but the family had a large house and room, and it was comfortable and comfortable. There were a few places that were reserved for the poor — a house for the poor, a place for the children and in the middle of it all. One of the houses was for the sick — the family paid for the food, the care of the children and other things. In the Bible, each of you was free, and in the Bible you had to pay for all the things your brother had to offer. When Jacob paid for the food, he did not take the land. You were well received.

---

### 🧠 Biology & Neuroscience

**Prompt:** *"The human brain is capable of"*

> The human brain is capable of many amazing things; here are just a few: The brain's sensory system can sense sound, movement, sight as well as touch, smell and taste. The ability to read and write is one of the most important things you can do to live a long and happy life without distractions. As we get older, our memory and language abilities tend to decline, making it harder for us to remember and concentrate. What Causes Alzheimer's Inability? The brain is not fully functioning well when we no longer have the ability to concentrate. The brain is not able to keep up with the demands of daily life, which is why memory and concentration are so important. As we age, we lose the capacity to focus, even when we…

---

> [!NOTE]
> Quality improves substantially as training continues toward 152,587 steps (~10B tokens). These samples are a snapshot at ~34% training — noticeably more coherent and topically consistent than the early ~7% checkpoint.

---

## 🏆 Benchmarking & Evaluation (`lm-evaluation-harness`)

Evaluate local Ultron checkpoints directly in-memory against standard online benchmarks (`HellaSwag`, `LAMBADA`, `PIQA`, `ARC-Easy`, `ARC-Challenge`) using EleutherAI's `lm-evaluation-harness`:

```bash
accelerate launch scripts/eval_lm_eval.py --checkpoint=accelerate_checkpoint --tasks=hellaswag,lambada_openai,piqa,arc_easy,arc_challenge
```

### Zero-Shot Benchmark Results

| Benchmark | Metric | Ultron (124M) Score | Original GPT-2 (124M) | Status |
| :--- | :--- | :---: | :---: | :---: |
| **PIQA** | Accuracy (`acc_norm`) | **68.00%** | ~62.80% | 🏆 **+5.20% vs GPT-2** |
| **HellaSwag** | Accuracy (`acc_norm`) | **45.00%** | ~33.70% | 🏆 **+11.30% vs GPT-2** |
| **LAMBADA** | Accuracy (`acc`) | **15.00%** | — | Mid-training baseline |
| **ARC-Easy** | Accuracy (`acc`) | **50.00%** | — | Mid-training baseline |
| **ARC-Challenge** | Accuracy (`acc`) | **10.00%** | — | Mid-training baseline |

---

## 🚀 Hugging Face Export & Hub Upload

Export local Accelerate checkpoints to Hugging Face format (`safetensors`, `config.json`, `hf_model.py`) and upload to Hugging Face Hub (`jaipkapoor99/ultron-124m`):

```bash
python3 scripts/export_to_hf.py
```

The export pipeline:

1. Executes unit tests (`tests/test_hf_wrapper.py`) to verify parameter pointer tying and logit parity.
2. Packages model weights, config, and `hf_model.py` into `hf_export/`.
3. Publishes model artifacts to Hugging Face Hub (`jaipkapoor99/ultron-124m`).

Once published, load Ultron directly using standard Hugging Face Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("jaipkapoor99/ultron-124m", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
```

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
