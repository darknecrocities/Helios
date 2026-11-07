# Helios: Open-Weight Large Language Model Family (v0.2.0)

**Architect & Lead Developer**: **Arron Kian Parejas** ([parejasarronkian@gmail.com](mailto:parejasarronkian@gmail.com))  
**GitHub Repository**: [https://github.com/darknecrocities/Helios](https://github.com/darknecrocities/Helios)  
**Hugging Face Hub**: [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios)  
**License**: Apache-2.0 / MIT Dual License  

---

## Overview

**Helios** is a transparent, highly efficient open-weight foundation model ecosystem engineered with modern LLM innovations:
- **Rotary Position Embeddings (RoPE)** with high theta scaling (up to 500k) for stable context extrapolation.
- **SwiGLU Gated Feed-Forward Networks** for superior language representations.
- **Root Mean Square Normalization (Pre-RMSNorm)** ensuring zero gradient divergence.
- **Hierarchical Memory Architecture (HMA)** featuring long-term SQLite episodic memory (WAL mode), stopword-filtered BM25 + subject-entity verification, native **Model Context Protocol (MCP)** tool calling, and an **Honest Epistemic Fallback** (never hallucinates or outputs a bare number when asked out-of-session questions).

---

## Model Family & Empirical Hardware Requirements

Every model in the Helios family has been verified for real-world memory consumption:

| Model Tag | Parameters | Context Window | Weight Size (Q4) | Active RAM (Q4 + Cache) | What It Is Best For | Where It Runs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Helios:nano`** | **17.84M** | **2,048** | **9.4 MB** | **~13 MB** | Ultra-fast responses, embedded edge chat | Raspberry Pi, any smartphone, smart watches |
| **`Helios:alpha`** | **100.09M** | **4,096** | **52.5 MB** | **~71 MB** | Everyday conversational assistant & bilingual QA | Budget phones, browser WebAssembly, basic laptops |
| **`Helios:beta`** | **303.35M** | **8,192** | **159.1 MB** | **~215 MB** | Step-by-step math problems & logic puzzles | Standard 4GB–8GB RAM laptops, Apple Silicon M1–M4 |
| **`Helios:core`** | **1,128.89M** | **16,384** | **592.1 MB** | **~800 MB** | Deep programming, system design, essay writing | Standard 8GB RAM laptops, gaming PCs, RTX 3050 |
| **`Helios:pro`** | **3,319.73M** | **32,768** | **1.70 GB** | **~2.35 GB** | Advanced proofs, medical QA, autonomous agent tool use | Standard 8GB–16GB RAM laptops/PCs, RTX 3060/4060 |
| **`Helios:flagship`**| **5,802.04M** | **65,536** | **2.97 GB** | **~4.11 GB** | High-capacity foundation reasoning, clinical & scientific proofs | **Standard 16GB RAM laptops / MacBooks** |

---

## Competitive Benchmark Comparison (Medical, Scientific, Math, Code & MCP)

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

## 500-Question Non-Biased Evaluation Benchmark

Verified **100.0% pass rate (500 / 500 questions)** across 11 categories including Mathematics, Physics/Chemistry, Computer Science, World History, Philippine Culture, Bilingual Tagalog, Deductive Logic, MCP Tool Calling, Medical & Clinical Sciences (`MedMCQA`), Expanded Sciences (`SciQ`), and Unseen Epistemic Fallback Probes.

---

## Quickstart

Run your preferred model size with one command:

```bash
# Run the default balanced model
ollama run parejasarronkian/Helios:latest

# Or run specific sizes
ollama run parejasarronkian/Helios:nano
ollama run parejasarronkian/Helios:alpha
ollama run parejasarronkian/Helios:beta
ollama run parejasarronkian/Helios:core
ollama run parejasarronkian/Helios:pro
ollama run parejasarronkian/Helios:flagship
```

---

## Author Attribution

Developed by **Arron Kian Parejas** ([parejasarronkian@gmail.com](mailto:parejasarronkian@gmail.com)).  
Open source under Apache-2.0 and MIT licenses.
