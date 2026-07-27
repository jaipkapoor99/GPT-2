# GPT-2 (124M 2026 SOTA) Pre-training Pipeline

A high-performance, modern PyTorch implementation of **GPT-2 (124M parameters)** pre-trained from scratch on the **Hugging Face FineWeb** dataset (`sample-10BT`), incorporating 2026 State-of-the-Art (SOTA) LLM training innovations.

Features **Rotary Position Embeddings (RoPE)**, **Muon Newton-Schulz Matrix Optimizer**, **Grouped-Query Attention (GQA)**, **SwiGLU FFN Activations**, **RMSNorm**, **100% Bias-Free Linear Layers**, **WSD (Warmup-Stable-Decay) Schedule**, **Zero-Copy Memory-Mapped Data Pipeline**, **FlashAttention-2 Integration**, **PyTorch 2.0 Graph Compilation**, and **Hugging Face Accelerate & Safetensors Integration**.

---

## 🌟 Key Features & Architectural Enhancements

- **Rotary Position Embeddings (RoPE):** Replaced static positional embeddings with RoPE (LLaMA 3 / Qwen 2.5 standard) applied to $Q$ and $K$ heads, enabling zero-shot context window extension.
- **Muon Newton-Schulz Matrix Optimizer:** Pre-training optimization using Keller Jordan's **Muon** (Momentum Orthogonalized by 5th-order Newton-Schulz iterations) for 2D body weights, paired with fused `AdamW` for 1D vectors/embeddings.
- **Grouped-Query Attention (GQA):** 12 Query heads and 4 Key/Value heads (3:1 Query-to-KV ratio), reducing KV-cache memory usage during autoregressive generation by **3x**.
- **Modernized SwiGLU FFN:** Replaced legacy GELU MLP with **SwiGLU Gated Linear Units** aligned to multiples of 64 for optimal GPU Tensor Core throughput.
- **RMSNorm & Bias-Free Layers:** Standardized on Root Mean Square Layer Normalization (RMSNorm) and removed linear projection bias parameters across all layers (`bias=False`).
- **Warmup-Stable-Decay (WSD) Schedule:** Replaced traditional cosine decay with WSD (80% stable phase, 20% cosine decay), allowing optimal checkpoint decay tuning.
- **Activation Checkpointing:** Optional gradient checkpointing (`--gradient-checkpointing`) reducing GPU VRAM memory consumption by **~60%**.
- **Zero-Copy Data Loader:** Memory-mapped disk slicing (`np.memmap`) for zero RAM allocation overhead during multi-billion token streaming.
- **Safetensors & Hugging Face Hub Integration:** Native export to `model.safetensors` format hosted live on Hugging Face Hub ([`jaipkapoor99/gpt2-2026-sota`](https://huggingface.co/jaipkapoor99/gpt2-2026-sota)).

---

## 📊 Pre-training Architecture & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Parameters** | 114,051,840 (114M) | GQA + Bias-Free SOTA 124M scale |
| **Layers / Query Heads / KV Heads** | 12 layers / 12 Q-heads / 4 KV-heads | GQA Transformer layout ($C=768, n_{head}=12, n_{kv\_head}=4$) |
| **Positional Embedding** | RoPE (Rotary) | Dynamic frequency base ($10,000$) |
| **FFN Activation** | SwiGLU | Multiples of 64 Tensor-Core aligned |
| **Normalization** | RMSNorm | $\epsilon=10^{-5}$ |
| **Context Window ($T$)** | 1,024 tokens | Extendable sequence length |
| **Micro-Batch Size ($B$)** | 16 | Per-GPU micro-batch size |
| **Gradient Accumulation** | 8 steps | Effective batch size = 128 sequences (131,072 tokens/step) |
| **Tokenizer** | SmolLM Vocab (49,152) | Efficient BPE tokenizer (`HuggingFaceTB/SmolLM-135M`) |
| **Precision** | BFloat16 (`bf16`) | Native mixed precision |
| **LR Schedule** | WSD | Warmup-Stable-Decay (80% stable, 20% decay) |
| **Optimizer** | Muon + AdamW | Newton-Schulz matrix optimizer + fused AdamW |

---

## 📂 Project Structure

```text
├── config.py             # Hyperparameters & GPT2Config dataclass
├── dataset.py            # Zero-Copy Memmap Sharded DataLoader
├── model.py              # GPT-2 Architecture (FlashAttention-2, SwiGLU FFN, Weight Tying)
├── train.py              # Distributed Pre-training Loop (Accelerate, bf16, torch.compile)
├── generate.py           # Custom PyTorch Autoregressive Text Generation CLI
├── test_hf_generate.py   # Hugging Face Transformers + Safetensors Generation CLI
├── upload_to_hf.py       # Automated Hugging Face Model Hub Uploader
├── tokenize_dataset.py   # FineWeb dataset tokenization into binary shards
├── muon.py               # Muon (Newton-Schulz Orthogonalized) Optimizer implementation
├── loss_curve.svg        # Pre-training loss curve vector visualization
├── loss_history.csv      # Step-by-step training and dev loss logs
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 📦 Quickstart: Using Pre-trained Weights from Hugging Face

### Option A: Using Hugging Face `transformers`

The pre-trained model weights are hosted on Hugging Face Model Hub in `safetensors` format:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "jaipkapoor99/gpt2-2026-sota"

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Option B: Running the Local CLI Test Script

Run our test script that automatically downloads the Safetensors checkpoint from Hugging Face and runs generation:

```bash
python test_hf_generate.py "Arrakis is a desert planet where" --max-tokens 100 --temperature 0.8
```

---

## 🏃 Running Pre-training Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Tokenize FineWeb Dataset
Tokenize the FineWeb dataset into compact 100M-token binary shards (`uint16` format):
```bash
python tokenize_dataset.py
```

### 3. Launch Distributed Pre-training
Start pre-training with PyTorch 2.0 compile and Hugging Face Accelerate:
```bash
accelerate launch train.py --max-steps 10000
```

### 4. Resume Pre-training with Accelerate
Resume full multi-GPU training state (model weights, optimizer state, LR schedule, and RNG state) seamlessly using Accelerate:
```bash
accelerate launch train.py --resume --max-steps 20000
```
Or specify a custom Accelerate checkpoint directory / PyTorch weights file:
```bash
accelerate launch train.py --load-checkpoint accelerate_checkpoint --max-steps 20000
```

### 5. Upload Checkpoint to Hugging Face Hub
To push your model weights (`model.safetensors`), configurations, and documentation directly to Hugging Face:
```bash
python upload_to_hf.py
```

---

## 📈 Visualizing Training Loss

Training logs are saved in real-time to `loss_history.json` and `loss_history.csv`. A vector graphic plot of the pre-training loss trajectory is rendered to [`loss_curve.svg`](file:///home/jaipkapoor99/Kaggle/Andrej%20Karpathy%20Course/GPT-2/loss_curve.svg).

![Loss Curve](loss_curve.svg)

---

## 📜 Acknowledgments
- Andrej Karpathy for the inspiring [*Neural Networks: Zero to Hero*](https://github.com/karpathy/build-nanogpt) course and `nanoGPT` project.
- Hugging Face for the [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset and `transformers` ecosystem.
