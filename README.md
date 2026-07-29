# GPT-2 (124M 2026 SOTA) Pre-training Pipeline

A high-performance, modern PyTorch implementation of **GPT-2 (124M parameters)** pre-trained from scratch on the **FineWeb-Edu** dataset, incorporating 2026 State-of-the-Art (SOTA) LLM training innovations.

Features **Rotary Position Embeddings (RoPE)**, **QK-Head RMSNorm**, **Muon Newton-Schulz Matrix Optimizer**, **Grouped-Query Attention (GQA)**, **SwiGLU FFN Activations**, **Logit Soft-Capping**, **100% Bias-Free Linear Layers**, **WSD (Warmup-Stable-Decay) Schedule**, **Zero-Copy Memory-Mapped Data Pipeline**, and **PyTorch 2.0 Graph Compilation**.

---

## 🌟 Key Features & Architectural Baseline

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
> **Hugging Face Component (TODO):** Hugging Face Hub uploads, pipeline wrappers, and remote model publishing are intentionally deferred as future **TODO** items. Current focus is strictly local-first pre-training and benchmarking.

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
GPT-2/
├── model.py              # PyTorch GPT-2 (RoPE + GQA + SwiGLU + RMSNorm + QKNorm + Logit SoftCap)
├── config.py             # Model & Hyperparameter Configuration Dataclass
├── dataset.py            # Zero-Copy Memmap Sharded Dataset Loader
├── train.py              # Main Distributed Accelerated Training Script
├── trainer.py            # Trainer Class with Keller Jordan Muon + AdamW
├── requirements.txt      # Dependencies
├── README.md             # Project Overview & Architecture Guide
├── tests/                # Unit & Integration Tests (Accelerate + torch.testing)
│   └── test_model.py
└── scripts/              # Helper Scripts
    └── tokenize_dataset.py # FineWeb-Edu dataset tokenization into binary shards
```

---

## 🚀 Quickstart & Workflow Guide

### 1. Installation

```bash
git clone https://github.com/jaipkapoor99/GPT-2.0.git
cd GPT-2.0
pip install -r requirements.txt
```

### 2. Tokenize Dataset

Tokenize the FineWeb-Edu dataset into compact binary shards:

```bash
python3 scripts/tokenize_dataset.py
```

### 3. Environment & Accelerate Configuration

Optionally configure Hugging Face Accelerate for your machine setup (`bf16` precision & `inductor` TorchDynamo backend):

```bash
accelerate config
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

All runs are tracked with **Weights & Biases** under the `gpt2-pretraining` project. The following metrics are logged at every step:

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
python3 -m unittest discover tests
```

---

## 🧪 Sample Generations (Step 10,750 / 152,587 — ~7% trained)

These samples were produced by `generate.py` at early training (≈7% through the full 10B-token run), temperature 0.85, top-k 50.

---

**Prompt:** *"The history of science shows that"*

> The history of science shows that it has been used successfully in several different fields, including the fields of Physics, Physics of Matter, Astronomy, and Cosmology. During the centuries since the beginning of the era, science has served to educate individuals about the world around them and to inform them about the complexities of their world. A great example of this type of knowledge is that scientists and mathematicians have been working to understand the cosmos in a vast, diverse way. Their understanding of the cosmos has helped shape the way they understand the world and how to understand it. However, this kind of knowledge has also contributed to the growth of the…

---

**Prompt:** *"The best way to learn programming is"*

> The best way to learn programming is through programming in the C .c and c programming languages. C .c programming languages have been developed for use with programming and can be used for programming a large number of languages, including C .c, C .etc. Most programming languages are designed for programming in C .c. The reason why you should learn C.c is to avoid any of the many problems that lie behind it. Python programming language is a great programming language and a good starting point in the C programming language. When…

---

**Prompt:** *"In the depths of the ocean"*

> In the depths of the ocean, he can now see that the ocean is the most diverse, with a lot of rocks and rocks. He can also see that the continents were more than a single continent, but they were made of different areas, which made up less than one percent of the Earth's mass. The last three centuries were the warmest years ever recorded on the planet. To me, this meant that the 19th century was the hottest year ever recorded on Earth. With the first ice age, more and more we live on the planet…

---

**Prompt:** *"Once upon a time in a kingdom far away"*

> Once upon a time in a kingdom far away from the centre, once upon a time in every thousand years, when God created every part of the earth, and only the one through the whole human race of heaven and earth, and all the parts of Christ, and all that was in Him, that all the earth, which had been given to them, could be made to one another, and that all things God made to Him made one another…

---

> [!NOTE]
> Quality improves substantially as training continues toward 152,587 steps (~10B tokens). These samples are a snapshot at early training to demonstrate the generation pipeline is working end-to-end.

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
