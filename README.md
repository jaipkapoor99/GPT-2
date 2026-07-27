# GPT-2 (124M) SOTA Pre-training Pipeline

A high-performance, modern PyTorch implementation of **GPT-2 (124M parameters)** pre-trained from scratch on the **Hugging Face FineWeb** dataset (`sample-10BT`), following Andrej Karpathy's *Neural Networks: Zero to Hero* course with 2026 State-of-the-Art (SOTA) LLM training techniques.

Features a **Zero-Copy Memory-Mapped Sharded Data Pipeline**, **SwiGLU FFN Activations**, **FlashAttention-2 Integration**, **PyTorch 2.0 Graph Compilation**, **`bf16` Mixed-Precision Acceleration**, and direct **Hugging Face Hub Integration (Safetensors)**.

---

## 🌟 Key Features & Architectural Enhancements

- **Modernized FFN Architecture:** Replaced legacy GELU MLP with **SwiGLU Gated Linear Units** (as used in LLaMA 3 & Qwen 2.5), dynamically aligned to multiples of 64 for optimal GPU Tensor Core throughput.
- **FlashAttention-2:** Leverages PyTorch's native `F.scaled_dot_product_attention` for $O(N)$ memory efficiency and high FLOP utilization.
- **Zero-Copy Data Loader:** Memory-mapped disk slicing (`np.memmap`) for zero RAM allocation overhead during multi-billion token streaming.
- **TorchInductor Graph Compilation:** Accelerated execution graph compilation via `torch.compile()` with high-precision MatMul math (`torch.set_float32_matmul_precision('high')`).
- **Distributed Training & Mixed Precision:** Built on Hugging Face `Accelerate` with native `bfloat16` precision and gradient accumulation.
- **Safetensors & Hugging Face Hub Integration:** Native export to `model.safetensors` format hosted live on Hugging Face Hub ([`jaipkapoor99/gpt2-2026-sota`](https://huggingface.co/jaipkapoor99/gpt2-2026-sota) & [`jaipkapoor99/gpt2-124m-fineweb`](https://huggingface.co/jaipkapoor99/gpt2-124m-fineweb)).
- **Advanced Optimization Options:** Includes standard AdamW ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.1$) as well as experimental support for the **Muon** (Momentum Orthogonalized by Newton-Schulz) optimizer (`muon.py`).

---

## 📊 Pre-training Architecture & Hyperparameters

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model Parameters** | 123,545,088 (124M) | Standard GPT-2 Small parameter scale |
| **Layers / Heads / Embed Dim** | 12 layers / 12 heads / 768 dim | Transformer core layout ($C=768, n_{head}=12$) |
| **Head Dimension** | 64 | Dimension per attention head |
| **Context Window ($T$)** | 1,024 tokens | Sequence length |
| **Micro-Batch Size ($B$)** | 16 | Per-GPU batch size |
| **Gradient Accumulation** | 8 steps | Effective batch size = 128 sequences (131,072 tokens/step) |
| **Tokenizer** | SmolLM Vocab (49,152) | Efficient BPE tokenizer (`HuggingFaceTB/SmolLM-135M`) |
| **Precision** | BFloat16 (`bf16`) | Native mixed precision |
| **Max Learning Rate** | $6 \times 10^{-4}$ | Cosine decay with 200 warmup steps |
| **Min Learning Rate** | $6 \times 10^{-5}$ | $10\%$ final decay floor |
| **Optimizer** | AdamW | $\beta_1=0.9, \beta_2=0.95$, weight decay = $0.1$ |

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
