---
language:
- en
license: mit
library_name: pytorch
tags:
- gpt2
- causal-lm
- flash-attention
- swiglu
- fineweb
- smollm
metrics:
- loss
pipeline_tag: text-generation
widget:
- text: "The future of artificial intelligence is"
- text: "In the modern era of deep learning,"
---

# GPT-2 (124M) 2026 SOTA PyTorch Model Card

A high-performance **Masterclass Showcase of AI Engineering Skill**—re-implementing the **GPT-2 (124M)** architecture upgraded with modern 2026 state-of-the-art Large Language Model components, zero-copy binary dataset streaming, and hardware-accelerated kernels.

* **Repository:** [https://huggingface.co/jaipkapoor99/gpt2-2026-sota](https://huggingface.co/jaipkapoor99/gpt2-2026-sota)
* **Architecture:** GPT-2 (124M) with SwiGLU Gated MLPs & FlashAttention-2
* **Dataset:** FineWeb 10BT Sample (`HuggingFaceFW/fineweb`)
* **Tokenizer:** SmolLM Byte-BPE (`HuggingFaceTB/SmolLM-135M` - 49,152 Vocab Size)

---

## Key Architectural & Technical Highlights

1. **FlashAttention-2 (`F.scaled_dot_product_attention`):** High-speed fused CUDA attention operating on 4D tensors `(B, n_head, T, head_dim)`.
2. **SwiGLU Gated FeedForward Network:** Modern Gated Swish activation mechanism used in LLaMA 3, Qwen 2.5, and Mistral ($\text{hidden\_dim} = \frac{8}{3} C$, rounded to multiple of 64).
3. **SmolLM Byte-BPE Tokenizer (`HuggingFaceTB/SmolLM-135M`):** 49,152 vocabulary size optimized for hardware matrix multiplications on NVIDIA Tensor Cores ($49,152 = 768 \times 64$).
4. **Weight Tying:** Shares parameters between token embeddings (`wte`) and output projection (`lm_head`).
5. **Zero-Copy Memory Mapped Dataset Loading (`np.memmap`):** Direct token offset slicing over 1.9 Billion tokens (`shards/fineweb_shard_XXXX.bin`) with 0.00 MB RAM overhead.
6. **Pre-trained Weights & Local Checkpoint Loader:** Flexible CLI flags to load pre-trained OpenAI weights (`--from-pretrained gpt2`) or resume local training (`--load-checkpoint gpt2_model.pth`).
7. **Loss Tabulation & Visualization:** Tabulates metric rows during training, exports `loss_history.csv` & `loss_history.json`, and renders a high-res `loss_curve.png`.

---

## Pre-Training Loss Progression (3,000 Steps / 393M Tokens)

```text
+-------+------------+-----------+-----------+
| Step  | Train Loss | Dev Loss  |    LR     |
+-------+------------+-----------+-----------+
|   250 |    5.7479  |   5.7740  | 6.00e-04  |
|   500 |    5.0730  |   4.9425  | 5.85e-04  |
|   750 |    4.6465  |   4.4856  | 5.50e-04  |
|  1000 |    4.3635  |   4.2718  | 4.99e-04  |
|  1250 |    4.2426  |   4.1470  | 4.34e-04  |
|  1500 |    4.0342  |   4.0543  | 3.61e-04  |
|  1750 |    3.8950  |   3.9819  | 2.85e-04  |
|  2000 |    4.0211  |   3.9269  | 2.13e-04  |
|  2250 |    3.8842  |   3.8822  | 1.50e-04  |
|  2500 |    3.9335  |   3.8500  | 1.02e-04  |
|  2750 |    3.8288  |   3.8276  | 7.06e-05  |
|  3000 |    3.8248  |   3.8123  | 6.00e-05  |
+-------+------------+-----------+-----------+
```

---

## Sample Model Output (Authentic 3,000-Step Checkpoint Generation)

**Prompt:** `"The future of artificial intelligence is"`

```text
The future of artificial intelligence is a good thing, but it is not that easy.
We’ve been in the industry for more than 10 years with the latest technology, and we’ve got a few keynotes.

The 100% increase in the average American economy has caused more than 30% of Americans to lose their jobs.
One of the most important things in life is
```

---

## System Hardware & Environment Configuration

This repository was benchmarked and trained on the following hardware & software environment:

### Hardware Specifications & Memory Footprint
* **CPU:** AMD Ryzen 7 9800X3D (8 Cores / 16 Threads with 3D V-Cache Technology)
* **GPU:** NVIDIA GeForce RTX 5090 (32GB VRAM, 600W Power Limit)
* **VRAM Consumption:** **~28 GB VRAM** active allocation under current pre-training configuration ($B = 16, T = 1024$, Grad Accum $= 8$).
* **GPU Performance:** ~576W active power draw, 98% compute utilization, peak 71°C thermal performance under full load.
* **Host Architecture:** WSL2 on Ubuntu 24.04 LTS (Linux Kernel 6.6+)

### Software & AI Stack
* **CUDA / Driver Version:** CUDA 13.3 / Driver Version 610.43.02
* **Python Runtime:** Python 3.13.11 (Miniconda 26.1.1)
* **PyTorch Ecosystem:** PyTorch 2.6+ (CUDA 12.8/13.3 enabled)
* **Hugging Face Stack:** `transformers` v5.14.1, `accelerate` v1.14.0, `datasets` v5.0.0

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

### 3. Launch Pre-training (~28 GB VRAM Peak)
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
python generate.py "The future of artificial intelligence is" --max-tokens 80
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
