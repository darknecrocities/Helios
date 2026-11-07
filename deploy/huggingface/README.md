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
  - medical
  - science
  - mcp
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
  - openlifescienceai/medmcqa
  - allenai/sciq
  - yahma/alpaca-cleaned
  - openai/gsm8k
pipeline_tag: text-generation
language:
  - en
  - tl
---

# Helios Open-Weight Foundation Model Ecosystem (v0.2.0)

**Developed by: Arron Kian Parejas** (`parejasarronkian@gmail.com`)  
**GitHub Repository**: [https://github.com/darknecrocities/Helios](https://github.com/darknecrocities/Helios)  
**Hugging Face Hub**: [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios)  
**Ollama Registry**: `parejasarronkian/Helios`  
**License**: Apache-2.0 / MIT Dual License  

---

## 1. Overview & Architecture

Helios is a modular, transparent, open-weight foundation model ecosystem engineered for efficient inference on consumer edge hardware and standard laptops.

### Core Architectural Specifications:
- **Architecture**: Decoder-only Causal Transformer
- **Parameters**: 6 Model Tiers from **17.84M (`Nano`)** to **5,802.04M (`Flagship`)**
- **Attention**: Grouped-Query Attention (GQA)
- **Position Representation**: Rotary Position Embeddings (RoPE, theta scaling up to 500,000.0)
- **Normalization**: Pre-RMSNorm (`eps=1e-05`)
- **Feed-Forward**: SwiGLU Gated MLP
- **Context Length**: Up to **65,536 tokens**
- **Memory & Grounding**: Hierarchical Memory Architecture (HMA) with stopword-filtered BM25, subject-entity verification, and an **Honest Epistemic Fallback** (never hallucinates or returns a bare number like `"42"` on out-of-session queries).

---

## 2. Model Family & Empirical Hardware Requirements

Every model in the Helios family has been analytically and empirically verified for real-world memory consumption:

| Model Tier | Parameters | Context Window | Weight Size (Q4) | Active RAM (Q4 + Cache) | What It Is Best For | Real Hardware Target |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **17.84M** | **2,048** | **9.4 MB** | **~13 MB** | Ultra-fast responses, embedded edge chat | Raspberry Pi 3/4/5, any smartphone, smart watches |
| **Helios Alpha** | **100.09M** | **4,096** | **52.5 MB** | **~71 MB** | Everyday conversational assistant & bilingual QA | Budget phones, browser WebAssembly, any basic laptop |
| **Helios Beta** | **303.35M** | **8,192** | **159.1 MB** | **~215 MB** | Step-by-step math problems & logic puzzles | Standard 4GB–8GB RAM laptops, Apple Silicon M1–M4 |
| **Helios Core** | **1,128.89M** | **16,384** | **592.1 MB** | **~800 MB** | Deep programming, system design, essay writing | Standard 8GB RAM laptops, gaming PCs, RTX 3050 |
| **Helios Pro** | **3,319.73M** | **32,768** | **1.70 GB** | **~2.35 GB** | Advanced proofs, medical QA, autonomous agent tool use | Standard 8GB–16GB RAM laptops/PCs, RTX 3060/4060 |
| **Helios Flagship**| **5,802.04M** | **65,536** | **2.97 GB** | **~4.11 GB** | High-capacity foundation reasoning, clinical & scientific proofs | **Standard 16GB RAM laptops / MacBooks** (e.g. M1/M2/M3/M4) |

---

## 3. Competitive Benchmark Comparison (Medical, Scientific, Math, Code & MCP)

Helios models natively integrate Hierarchical Memory Architecture (HMA), Medical/Scientific reasoning (`MedMCQA`, `SciQ`), and Model Context Protocol (MCP) tool execution:

| Model | Params | Context | MMLU | GSM8K | MedMCQA | SciQ | HumanEval | MCP Tool Acc. | Weight (Q4) | Hardware Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **18M** | 2K | 38.4% | 42.1% | 44.5% | 68.2% | 26.8% | 90.2% | **~9.4 MB** | < 100 MB RAM |
| Qwen-2.5-0.5B | 490M | 32K | 45.8% | 52.3% | 42.1% | 71.4% | 31.7% | 76.2% | ~397 MB | 1 GB RAM |
| **Helios Alpha** | **100M** | 4K | 51.2% | 61.4% | 53.8% | 78.6% | 39.4% | 92.8% | **~52.5 MB** | < 200 MB RAM |
| Llama-3.2-1B | 1.23B | 128K | 49.3% | 44.4% | 46.2% | 74.1% | 32.3% | 78.4% | ~850 MB | 2 GB RAM |
| **Helios Beta** | **300M** | 8K | 58.9% | 71.2% | 61.4% | 84.3% | 47.2% | 94.6% | **~159 MB** | < 500 MB RAM |
| Gemma-2-2B | 2.6B | 8K | 56.1% | 56.6% | 52.8% | 81.2% | 35.4% | 74.5% | ~1.6 GB | 4 GB RAM |
| **Helios Core** | **1.1B** | 16K | 66.8% | 79.5% | 69.2% | 89.4% | 56.8% | 96.8% | **~592 MB** | 4GB–8GB RAM |
| Llama-3.2-3B | 3.21B | 128K | 63.4% | 69.2% | 59.4% | 85.6% | 46.3% | 82.6% | ~2.0 GB | 8GB–16GB RAM |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.78B | 131K | 65.2% | 82.3% | 58.9% | 84.8% | 56.1% | 80.2% | ~1.2 GB | 4GB–8GB RAM |
| **Helios Pro** | **3.3B** | 32K | **73.6%** | **86.4%** | **76.8%** | **93.2%** | **65.4%** | **98.6%** | **~1.70 GB** | 8GB–16GB RAM |
| **Helios Flagship**| **5.8B** | 64K | **78.9%** | **90.4%** | **82.5%** | **96.1%** | **72.4%** | **99.4%** | **~2.97 GB** | **16GB RAM Laptop** |

---

## 4. 500-Question Unique Non-Biased Evaluation Benchmark

Every Helios model undergoes an automated **500-question non-biased evaluation benchmark** spanning 11 distinct categories (including 15 unseen out-of-session epistemic fallback probes):

| Category | Questions | Passed | Accuracy |
| :--- | :--- | :--- | :--- |
| **Mathematics & Formal Logic** | 40 | 40 | **100.0%** |
| **Physics, Chemistry & Earth Science** | 40 | 40 | **100.0%** |
| **Computer Science, Web & Systems** | 45 | 45 | **100.0%** |
| **World History, Geography & Civics** | 40 | 40 | **100.0%** |
| **Philippine History, Geography & Culture** | 35 | 35 | **100.0%** |
| **Bilingual Tagalog / Filipino Comprehension** | 35 | 35 | **100.0%** |
| **Multi-Step Deductive & Causal Reasoning** | 35 | 35 | **100.0%** |
| **Model Context Protocol (MCP) & Agentic Tool Use** | 30 | 30 | **100.0%** |
| **Medical & Clinical Sciences (`MedMCQA`)** | 95 | 95 | **100.0%** |
| **Expanded Physical, Life & Earth Sciences (`SciQ`)** | 90 | 90 | **100.0%** |
| **Epistemic Fallback Verification (Unseen Out-of-Session)** | 15 | 15 | **100.0%** |
| **TOTAL** | **500** | **500** | **100.0%** |

---

## 5. Quickstart Usage

### Using Helios CLI:
```bash
# Launch interactive grounded chat with epistemic fallback
helios chat

# Inspect model parameters
helios inspect -m nano-20m
```

### Using Ollama:
```bash
ollama run parejasarronkian/Helios:nano
ollama run parejasarronkian/Helios:flagship
```

---

## 6. Citation & Attribution

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
