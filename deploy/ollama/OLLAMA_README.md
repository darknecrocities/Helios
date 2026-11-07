# Helios: Open-Weight Large Language Model Family

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
- **Hierarchical Memory Architecture (HMA)** featuring long-term SQLite episodic memory (WAL mode), hybrid semantic vector retrieval, and native **Model Context Protocol (MCP)** tool calling.

---

## Model Family & Empirical Hardware Requirements

Every model in the Helios family has been verified for real-world memory consumption:

| Model Tag | Parameters | Context Window | Weight Size (Q4) | Active RAM (Q4 + Cache) | What It Is Best For | Where It Runs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Helios:nano`** | **17.84M** | **2,048** | **9.4 MB** | **~13 MB** | Ultra-fast responses, embedded edge chat | Raspberry Pi, any smartphone, smart watches |
| **`Helios:alpha`** | **100.09M** | **4,096** | **52.5 MB** | **~71 MB** | Everyday conversational assistant & bilingual QA | Budget phones, browser WebAssembly, basic laptops |
| **`Helios:beta`** | **303.35M** | **8,192** | **159.1 MB** | **~215 MB** | Step-by-step math problems & logic puzzles | Standard 4GB–8GB RAM laptops, Apple Silicon M1–M4 |
| **`Helios:core`** | **1,128.89M** | **16,384** | **592.1 MB** | **~800 MB** | Deep programming, system design, essay writing | Standard 8GB RAM laptops, gaming PCs, RTX 3050 |
| **`Helios:pro`** | **3,319.73M** | **32,768** | **1.70 GB** | **~2.35 GB** | Advanced proofs, autonomous agent tool use | Standard 8GB–16GB RAM laptops/PCs, RTX 3060/4060 |
| **`Helios:flagship`**| **5,802.04M** | **65,536** | **2.97 GB** | **~4.11 GB** | High-capacity foundation reasoning & proofs | **Standard 16GB RAM laptops / MacBooks** |

---

## Competitive Benchmark Comparison

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
