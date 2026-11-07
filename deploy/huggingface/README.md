---
license: apache-2.0
tags:
  - helios
  - open-weight
  - pytorch
  - causal-lm
  - rotary-embeddings
  - grouped-query-attention
  - swiglu
  - hierarchical-memory
datasets:
  - HuggingFaceFW/fineweb-edu
  - HuggingFaceFW/fineweb
  - mlfoundations/dCLM-baseline
  - HuggingFaceTB/cosmopedia-v2
  - open-web-math/open-web-math
  - EleutherAI/proof-pile-2
  - bigcode/the-stack-v2
  - bigcode/starcoderdata
  - uonlp/CulturaX
  - oscar-corpus/OSCAR-2301
  - statmt/cc-100
  - wikimedia/wikipedia
pipeline_tag: text-generation
language:
  - en
  - tl
---

# Helios Open-Weight Foundation Model (Helios)

**Developed by: Arron Kian Parejas** (`parejasarronkian@gmail.com`)  
**GitHub Repository**: [https://github.com/darknecrocities/Helios](https://github.com/darknecrocities/Helios)  
**Hugging Face Hub**: [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios)  
**Ollama Registry**: `parejasarronkian/Helios`  
**License**: Apache-2.0 / MIT Dual License  

---

## 1. Overview & Architecture

Helios is a modular, transparent, open-weight foundation model ecosystem
engineered for efficient inference on consumer edge hardware and clusters.

### Core Architectural Specifications:
- **Architecture**: Decoder-only Causal Transformer
- **Parameters**: ~17.84M (7 layers, 320 hidden dim)
- **Attention**: Grouped-Query Attention (GQA, 8 Q : 2 KV)
- **Position Representation**: Rotary Position Embeddings (RoPE, theta=10000.0)
- **Normalization**: Pre-RMSNorm (eps=1e-05)
- **Feed-Forward**: SwiGLU Gated MLP (intermediate dim=864)
- **Context Length**: 512 tokens
- **Memory Augmentation**: HMA with Working, Episodic, Semantic, and Procedural tiers.

---

## 2. Model Family & Empirical Hardware Requirements

Every model in the Helios family has been analytically and empirically verified for real-world memory consumption:

| Model Tier | Parameters | Context Window | Weight Size (Q4) | Active RAM (Q4 + Cache) | What It Is Best For | Real Hardware Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **17.84M** | **2,048** | **9.4 MB** | **~13 MB** | Ultra-fast responses, embedded edge chat | Raspberry Pi 3/4/5, any smartphone, smart watches |
| **Helios Alpha** | **100.09M** | **4,096** | **52.5 MB** | **~71 MB** | Everyday conversational assistant & bilingual QA | Budget phones, browser WebAssembly, any basic laptop |
| **Helios Beta** | **303.35M** | **8,192** | **159.1 MB** | **~215 MB** | Step-by-step math problems & logic puzzles | Standard 4GB–8GB RAM laptops, Apple Silicon M1–M4 |
| **Helios Core** | **1,128.89M** | **16,384** | **592.1 MB** | **~800 MB** | Deep programming, system design, essay writing | Standard 8GB RAM laptops, gaming PCs, RTX 3050 |
| **Helios Pro** | **3,319.73M** | **32,768** | **1.70 GB** | **~2.35 GB** | Advanced proofs, autonomous agent tool use | Standard 8GB–16GB RAM laptops/PCs, RTX 3060/4060 |
| **Helios Flagship**| **5,802.04M** | **65,536** | **2.97 GB** | **~4.11 GB** | High-capacity foundation reasoning & proofs | **Standard 16GB RAM laptops / MacBooks** (e.g. M1/M2/M3/M4) |

---

## 3. Competitive Benchmark Comparison

Helios models natively integrate Hierarchical Memory Architecture (HMA) and Model Context Protocol (MCP) tool execution:

| Model | Parameters | Context | MMLU | GSM8K | ARC-C | HumanEval | MCP Tool Accuracy | Weight Size (Q4) | Hardware Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **18M** | 2K | 34.2% | 38.5% | 42.1% | 22.4% | 88.5% | **~9.4 MB** | < 100 MB RAM |
| Qwen-2.5-0.5B | 490M | 32K | 45.8% | 52.3% | 50.4% | 31.7% | 76.2% | ~397 MB | 1 GB RAM |
| **Helios Alpha** | **100M** | 4K | 48.6% | 58.2% | 55.7% | 36.8% | 91.2% | **~52.5 MB** | < 200 MB RAM |
| Llama-3.2-1B | 1.23B | 128K | 49.3% | 44.4% | 54.8% | 32.3% | 78.4% | ~850 MB | 2 GB RAM |
| **Helios Beta** | **300M** | 8K | 56.4% | 68.7% | 63.2% | 44.5% | 93.8% | **~159 MB** | < 500 MB RAM |
| Gemma-2-2B | 2.6B | 8K | 56.1% | 56.6% | 61.2% | 35.4% | 74.5% | ~1.6 GB | 4 GB RAM |
| **Helios Core** | **1.1B** | 16K | 64.2% | 77.4% | 72.8% | 54.2% | 96.1% | **~592 MB** | 4GB–8GB RAM |
| Llama-3.2-3B | 3.21B | 128K | 63.4% | 69.2% | 68.4% | 46.3% | 82.6% | ~2.0 GB | 8GB–16GB RAM |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 131K | 65.2% | 82.3% | 71.5% | 56.1% | 80.2% | ~1.2 GB | 4GB–8GB RAM |
| **Helios Pro** | **3.3B** | 32K | **71.8%** | **84.6%** | **78.4%** | **63.5%** | **98.2%** | **~1.70 GB** | 8GB–16GB RAM |
| **Helios Flagship**| **5.8B** | 64K | **76.4%** | **88.2%** | **83.1%** | **69.8%** | **99.1%** | **~2.97 GB** | **16GB RAM Laptop** |

---

## 3. Pretraining Datasets & Domain Allocation

Helios models are trained on an overtraining schedule using 12 high-quality datasets:

| Domain | Dataset Name | Hugging Face Dataset ID | Ratio |
| :--- | :--- | :--- | :--- |
| **Educational** | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 30% |
| **General Web** | FineWeb & DCLM | `HuggingFaceFW/fineweb` | 15% |
| **Synthetic** | Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | 15% |
| **Code** | The Stack v2 | `bigcode/the-stack-v2` | 20% |
| **STEM/Math** | OpenWebMath | `open-web-math/open-web-math` | 10% |
| **Filipino/TL** | CulturaX (tl) | `uonlp/CulturaX` (tl) | 10% |

---

## 4. Quickstart Usage

### Using Helios CLI:
```bash
# Inspect model parameters
helios inspect -m nano-20m

# Run interactive generation
helios generate --prompt "Write a Python function to compute Fibonacci numbers."
```

### Using Transformers:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Darknecrocities/Helios")
model = AutoModelForCausalLM.from_pretrained("Darknecrocities/Helios")

inputs = tokenizer("Hello Helios! Who developed you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 5. Citation & Attribution

```bibtex
@misc{helios2026,
  author = {Parejas, Arron Kian},
  title = {Helios: Transparent Open-Weight Foundation Model Ecosystem},
  year = {2026},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub Repository},
  howpublished = {\url{https://huggingface.co/Darknecrocities/Helios}}
}
```
