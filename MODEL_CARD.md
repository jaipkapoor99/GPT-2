---
language:
- en
license: apache-2.0
tags:
- gpt2
- causal-lm
- text-generation
- SOTA-2026
- rotary-position-embeddings
- muon-optimizer
- swiglu
- grouped-query-attention
- flash-attention
datasets:
- HuggingFaceFW/fineweb
metrics:
- accuracy
- perplexity
pipeline_tag: text-generation
library_name: transformers
model-index:
- name: jaipkapoor99/gpt2-2026-sota
  results:
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: HellaSwag
      type: hellaswag
    metrics:
    - name: Accuracy (Normalized)
      type: acc_norm
      value: 0.2661
  - task:
      type: text-generation
      name: Text Generation
    dataset:
      name: ARC Easy
      type: arc_easy
    metrics:
    - name: Accuracy (Normalized)
      type: acc_norm
      value: 0.2689
---

# GPT-2 (124M 2026 SOTA) Model Card

## Model Details
- **Model Name:** `jaipkapoor99/gpt2-2026-sota`
- **Architecture:** GPT-2 (124M Parameters) with 2026 SOTA LLM Innovations
  - **Rotary Position Embeddings (RoPE)**
  - **Grouped-Query Attention (GQA)** (12 query heads, 4 KV heads)
  - **SwiGLU Activation Function**
  - **RMSNorm & Bias-Free Linear Layers**
- **Tokenizer:** SmolLM / LLaMA 3 BPE Vocabulary (`49,152` size)
- **Pre-training Dataset:** Hugging Face FineWeb (`sample-10BT`)

## Evaluation Results
- **Validation Loss:** `3.2869`
- **Validation Perplexity:** `26.76`
- **HellaSwag (acc_norm):** `0.2661`
- **ARC-Easy (acc_norm):** `0.2689`
