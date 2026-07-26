# GPT-2 (124M) SOTA PyTorch Showcase (2026 Edition)

A high-performance **Masterclass Showcase of AI Engineering Skill**—implementing the **GPT-2 (124M)** architecture upgraded with modern 2026 state-of-the-art Large Language Model components, zero-copy binary streaming, and hardware-accelerated kernels.

---

## System Hardware & Environment Configuration

This repository is benchmarked and optimized as a high-throughput showcase on the following hardware & software environment:

### Hardware Specifications
* **CPU:** AMD Ryzen 7 9800X3D (8 Cores / 16 Threads with 3D V-Cache Technology)
* **GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM, 600W Power Limit)
* **GPU Performance:** ~576W active power draw, 98% compute utilization, peak 71°C thermal performance under full load.
* **Host Architecture:** WSL2 on Ubuntu 24.04 LTS (Linux Kernel 6.6+)

### Software & AI Stack
* **CUDA / Driver Version:** CUDA 13.3 / Driver Version 610.43.02
* **Python Runtime:** Python 3.13.11 (Miniconda 26.1.1)
* **PyTorch Ecosystem:** PyTorch 2.6+ (CUDA 12.8/13.3 enabled)
* **Hugging Face Stack:** `transformers` v5.14.1, `accelerate` v1.14.0, `datasets` v5.0.0

---

## Architectural & Technical Highlights

1. **FlashAttention-2 (`F.scaled_dot_product_attention`):** High-speed fused CUDA attention operating on 4D tensors `(B, n_head, T, head_dim)`.
2. **SwiGLU Gated FeedForward Network:** Modern Gated Swish activation mechanism used in LLaMA 3, Qwen 2.5, and Mistral ($\text{hidden\_dim} = \frac{8}{3} C$, rounded to multiple of 64).
3. **SmolLM Byte-BPE Tokenizer (`HuggingFaceTB/SmolLM-135M`):** 49,152 vocabulary size optimized for hardware matrix multiplications on NVIDIA Tensor Cores ($49,152 = 768 \times 64$).
4. **Weight Tying:** Shares parameters between token embeddings (`wte`) and output projection (`lm_head`).
5. **Zero-Copy Memory Mapped Dataset Loading (`np.memmap`):** Direct token offset slicing over 1.9 Billion tokens (`shards/fineweb_shard_XXXX.bin`) with 0.00 MB RAM overhead.
6. **Pre-trained Weights & Local Checkpoint Loader:** Flexible CLI flags to load pre-trained OpenAI weights (`--from-pretrained gpt2`) or resume local training (`--load-checkpoint gpt2_model.pth`).
7. **Loss Tabulation & Visualization:** Tabulates metric rows during training, exports `loss_history.csv` & `loss_history.json`, and renders a high-res `loss_curve.png`.

---

## Directory Structure

```
GPT-2/
├── config.py             # GPT2Config dataclass (Dynamic vocab_size & head_dim)
├── model.py              # CausalSelfAttention, SwiGLUMLP, Block, & GPT2 model class
├── dataset.py            # Zero-Copy memmap binary dataset loader
├── tokenize_dataset.py   # Standalone 10B FineWeb dataset streaming pre-tokenization script
├── train.py              # Pre-training CLI script
├── generate.py           # Top-K Autoregressive text generation CLI
├── requirements.txt      # Project dependencies
└── README.md             # Project & System Documentation
```

---

## Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Pre-tokenize Dataset Shards
Stream FineWeb documents into 100M token binary shards:
```bash
python tokenize_dataset.py
```

### 3. Launch Pre-training
Train GPT-2 (124M) from scratch:
```bash
python train.py --max-steps 3000
```

Optionally load pre-trained OpenAI weights:
```bash
python train.py --from-pretrained gpt2
```

Optionally load local checkpoint weights:
```bash
python train.py --load-checkpoint gpt2_model.pth
```

### 4. Text Generation
Generate text using Top-K autoregressive sampling:
```bash
python generate.py "The future of artificial intelligence is" --max-tokens 60
```

---

## Detailed Parameter Breakdown (123,545,088 Total)

```text
===========================================
  GPT-2 (124M) DETAILED PARAMETER BREAKDOWN
===========================================
  Token Embeddings (wte)    : 37,748,736 (30.6%)
  Position Embeddings (wpe) : 786,432 (0.6%)
  12 Transformer Blocks (h) : 85,008,384 (68.8%)
  Final LayerNorm (ln_f)    : 1,536 (<0.1%)
  Output LM Head (lm_head)  : 0 (Weight Tied with wte)
-------------------------------------------
  TOTAL TRAINABLE PARAMETERS: 123,545,088
===========================================
```

---

## License & Acknowledgments
Built in alignment with modern 2026 AI engineering standards, drawing architectural inspiration from Andrej Karpathy's `nanoGPT` / `nanochat`, OpenAI GPT-2, LLaMA 3, and Hugging Face SmolLM.
