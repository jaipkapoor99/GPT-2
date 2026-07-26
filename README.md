# GPT-2 (124M) PyTorch Reproduction (2026 SOTA Standards)

A modular, production-grade PyTorch implementation of the **GPT-2 (124M)** architecture upgraded with modern 2026 state-of-the-art Large Language Model components.

---

## Key Architectural & CLI Features

1. **FlashAttention-2 (`F.scaled_dot_product_attention`):** High-speed fused CUDA attention operating on 4D tensors `(B, n_head, T, head_dim)`.
2. **SwiGLU Gated FeedForward Network:** Modern Gated Swish activation mechanism used in LLaMA 3, Qwen 2.5, and Mistral.
3. **SmolLM Byte-BPE Tokenizer (`HuggingFaceTB/SmolLM-135M`):** 49,152 vocabulary size optimized for hardware matrix multiplications on NVIDIA Tensor Cores ($49,152 = 768 \times 64$).
4. **Weight Tying:** Shares parameters between token embeddings (`wte`) and output projection (`lm_head`).
5. **Zero-RAM Memory Mapped Dataset Loading (`np.memmap`):** Pre-tokenizes text into compact `uint16` binary shards (`shards/fineweb_shard_XXXX.bin`) and streams batches directly from disk into GPU VRAM.
6. **Pre-trained Weights & Local Checkpoint Loader:** Flexible CLI flags to load pre-trained OpenAI weights (`--from-pretrained gpt2`) or resume local training (`--load-checkpoint gpt2_model.pth`).
7. **Loss Tabulation & Visualization:** Tabulates metric rows during training, exports `loss_history.csv` & `loss_history.json`, and renders a high-res `loss_curve.png`.

---

## Directory Structure

```
GPT-2/
├── config.py             # GPT2Config dataclass
├── model.py              # CausalSelfAttention, SwiGLUMLP, Block, & GPT2 model class
├── dataset.py            # Zero-RAM memmap binary dataset loader
├── tokenize_dataset.py   # Standalone 10B FineWeb dataset streaming pre-tokenization script
├── train.py              # Production training CLI script
├── generate.py           # Top-K Autoregressive text generation CLI
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
