# GPT-2 (124M) PyTorch Reproduction (2026 SOTA Standards)

A modular, production-grade PyTorch implementation of the **GPT-2 (124M)** architecture upgraded with modern 2026 state-of-the-art Large Language Model components.

---

## Key Architectural Enhancements

1. **FlashAttention-2 (`F.scaled_dot_product_attention`):** High-speed fused CUDA attention kernel operating on 4D tensors `(B, n_head, T, head_dim)`.
2. **SwiGLU Gated FeedForward Network:** Replaces 2019 GELU with the modern Gated Swish activation mechanism used in LLaMA 3, Qwen 2.5, and Mistral.
3. **SmolLM Byte-BPE Tokenizer (`HuggingFaceTB/SmolLM-135M`):** 49,152 vocabulary size optimized for hardware matrix multiplications on NVIDIA Tensor Cores ($49,152 = 768 \times 64$).
4. **Weight Tying:** Shares parameters between token embeddings (`wte`) and output projection (`lm_head`).
5. **Zero-RAM Memory Mapped Dataset Loading (`np.memmap`):** Pre-tokenizes text into compact `uint16` binary shards (`shards/fineweb_shard_XXXX.bin`) and streams batches directly from disk into GPU VRAM.
6. **Hugging Face `accelerate` & Gradient Accumulation:** Automated FP16 mixed precision and high-throughput batching.

---

## Directory Structure

```
GPT-2/
├── config.py             # GPT2Config dataclass
├── model.py              # CausalSelfAttention, SwiGLUMLP, Block, & GPT2 PyTorch model
├── dataset.py            # Zero-RAM memmap binary dataset loader
├── tokenize_dataset.py   # Standalone 10B FineWeb dataset streaming pre-tokenization script
├── train.py              # Distributed / HF Accelerate training script
├── generate.py           # Top-K Autoregressive text generation CLI
├── GPT2.ipynb            # Interactive Jupyter notebook
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
```

---

## Quickstart

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
Train GPT-2 (124M) using Hugging Face Accelerate:
```bash
python train.py
```

### 4. Text Generation
Generate text using Top-K autoregressive sampling:
```bash
python generate.py "Hello, I am a language model,"
```

---

## Parameter Breakdown (123,591,168 Total)

```text
===========================================
  GPT-2 (124M) DETAILED PARAMETER BREAKDOWN
===========================================
  Token Embeddings (wte)    : 37,748,736 (30.5%)
  Position Embeddings (wpe) : 786,432 (0.6%)
  12 Transformer Blocks (h) : 85,054,464 (68.8%)
  Final LayerNorm (ln_f)    : 1,536 (<0.1%)
  Output LM Head (lm_head)  : 0 (Weight Tied with wte)
-------------------------------------------
  TOTAL TRAINABLE PARAMETERS: 123,591,168
===========================================
```
