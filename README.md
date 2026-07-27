# GPT-2 (124M) Pre-training Pipeline from Scratch

A high-performance PyTorch pre-training implementation of **GPT-2 (124M parameters)** trained on the **FineWeb-Edu** dataset from scratch, following Andrej Karpathy's *Neural Networks: Zero to Hero* course.

Features a **Zero-Copy Memory-Mapped Dataset pipeline**, **Hugging Face Accelerate integration with `bf16` mixed precision**, and **fused FlashAttention kernels**.

---

## 🚀 Key Features & Performance
- **Zero-Copy Data Loader:** Direct memory-mapped disk slicing (`np.memmap`) for zero RAM allocation overhead.
- **Hardware Acceleration:** Native PyTorch `bfloat16` precision and FlashAttention-2 integration optimized for NVIDIA RTX 50-series (Blackwell) GPUs.
- **Hugging Face Hub Integration:** Direct checkpoint conversion and hosting on Hugging Face Hub (`jaipkapoor99/gpt2-124m-fineweb`).
- **Optimization Strategy:** Cosine learning rate decay with linear warmup, AdamW ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$), and gradient norm clipping.

---

## 📊 Pre-training Architecture & Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Model Size** | 123,545,088 parameters (124M) |
| **Layers / Heads / Embed Dim** | 12 layers / 12 heads / 768 dim |
| **Context Window ($T$)** | 1,024 tokens |
| **Micro-Batch Size ($B$)** | 16 |
| **Gradient Accumulation** | 8 steps (Effective Batch = 128 sequences / 131,072 tokens per step) |
| **Precision** | BFloat16 (`bf16`) |
| **Dropout** | 0.0 |
| **Max Learning Rate** | $6 \times 10^{-4}$ |
| **Min Learning Rate** | $6 \times 10^{-5}$ |
| **Warmup Steps** | 200 |

---

## 🛠️ Project Structure

```text
├── config.py             # Hyperparameters & GPT2Config dataclass
├── dataset.py            # Zero-Copy Memmap Sharded DataLoader
├── model.py              # GPT-2 Architecture (Attention, MLP, Weight Tying)
├── train.py              # Main Distributed Pre-training Loop (Accelerate, Logging, Evaluation)
├── tokenize_dataset.py   # FineWeb-Edu tokenization into binary shards
├── .gitignore            # Clean git exclusion rules
└── README.md             # Project documentation
```

---

## 📦 Using Pre-trained Weights from Hugging Face

The trained model checkpoint is available on the Hugging Face Hub:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_id = "jaipkapoor99/gpt2-124m-fineweb"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained(model_id)

prompt = "Ancient Rome was"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=50, top_p=0.95)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🏃 Running Training Locally

1. **Install Dependencies:**
   ```bash
   pip install torch transformers accelerate datasets numpy matplotlib tqdm
   ```

2. **Tokenize Dataset:**
   ```bash
   python tokenize_dataset.py
   ```

3. **Start Pre-training:**
   ```bash
   python train.py --max-steps 10000
   ```
