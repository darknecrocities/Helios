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

**Developed by: Arron Kian Parejas** (`arronkianparejas@gmail.com`)  
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

## 2. Parameter Scaling Roadmap & Exit Gates

| Stage | Target Size | Hardware Target | Status | Exit Gates |
| :--- | :--- | :--- | :--- | :--- |
| **Nano** | 10M–40M | Apple Silicon M3 | **Active** | Deterministic resume, overfit tiny corpus |
| **Alpha** | 100M | Single GPU / MPS | Planned | Perplexity < 25, fertility validated |
| **Beta** | 300M | Single GPU (24GB) | Planned | GQA KV-cache efficiency, zero regressions |
| **Core** | 1B | Multi-GPU Node | Roadmap | FSDP2 sharding, cluster recipe |
| **Pro** | 3B | Cluster (8x H100) | Long-Term | Multi-node recovery, optimizer sharding |
| **Flagship**| 6B | Compute Cluster | Long-Term | Full training provenance, safety review |

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
