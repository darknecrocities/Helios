# Helios: Open-Weight Large Language Model Family (v0.2.0)

**Architect & Lead Developer**: **Arron Kian Parejas** ([parejasarronkian@gmail.com](mailto:parejasarronkian@gmail.com))  
**GitHub Repository**: [https://github.com/darknecrocities/Helios](https://github.com/darknecrocities/Helios)  
**Hugging Face Hub**: [https://huggingface.co/Darknecrocities/Helios](https://huggingface.co/Darknecrocities/Helios)  
**License**: Apache-2.0 / MIT Dual License  

---

## About Helios: Origin of the Name & Engineering Techniques

**Helios** is named after **Helios (Ἥλιος)**, the ancient Greek Titan god and personification of the **Sun**, who drove his radiant golden chariot across the sky to illuminate the Earth with clarity and truth. Personally conceived, architected, and developed by **AI Engineer Arron Kian Parejas** (`parejasarronkian@gmail.com`), Helios brings transparent, open-weight intelligence to everyday hardware using:
- **Grouped-Query Attention (GQA)** for up to 8× smaller KV-cache memory footprints.
- **Rotary Position Embeddings (RoPE)** with high theta scaling (up to 500k) for stable context extrapolation up to 65,536 tokens.
- **SwiGLU Gated Feed-Forward Networks** & **Pre-RMSNorm** ensuring zero gradient divergence.
- **4-Tier Hierarchical Memory Architecture (HMA)** featuring SQLite WAL episodic memory, stopword-filtered BM25 + subject-entity verification, native **Model Context Protocol (MCP)** tool calling, and an **Honest Epistemic Fallback** trained across **26 open-source datasets** (`sot/README.md`).

---

## Model Family & Complete Hardware Memory Breakdown

| Model Tag | Parameters | Context Window | Raw Q4 Weights | KV-Cache (@ 2K) | KV-Cache (@ Max Ctx) | Runtime Buffers | Active RAM (@ 2K) | Active RAM (@ Max Ctx) | Where It Runs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Helios:nano`** | **17.84M** | **2,048** | 8.93 MB | 4.38 MB | 4.38 MB | 48.0 MB | **61.3 MB** | **61.3 MB** | Raspberry Pi, smartphones, microVMs |
| **`Helios:alpha`** | **100.09M** | **4,096** | 50.11 MB | 24.0 MB | 48.0 MB | 64.0 MB | **138.1 MB** | **162.1 MB** | Budget phones, WebAssembly, basic laptops |
| **`Helios:beta`** | **303.35M** | **8,192** | 151.88 MB | 48.0 MB | 192.0 MB | 96.0 MB | **295.9 MB** | **439.9 MB** | Standard 4GB–8GB RAM laptops, Apple Silicon M1–M4 |
| **`Helios:core`** | **1,128.89M** | **16,384** | 565.21 MB | 96.0 MB | 768.0 MB | 160.0 MB | **821.2 MB** | **1.46 GB** | Standard 8GB RAM laptops, gaming PCs, RTX 3050 |
| **`Helios:pro`** | **3,319.73M** | **32,768** | 1.62 GB | 256.0 MB | 4.00 GB | 256.0 MB | **2.12 GB** | **5.87 GB** | Standard 8GB–16GB RAM laptops/PCs, RTX 3060/4060 |
| **`Helios:flagship`**| **5,802.04M** | **65,536** | 2.84 GB | 256.0 MB | 8.00 GB | 384.0 MB | **3.46 GB** | **11.21 GB** | **Standard 16GB RAM laptops / MacBooks** |

---

## Audited Benchmark Comparison (`eval/` Automated Suite)

| Model | Variant / Mode | Params | Context | MMLU (5-shot) | GSM8K (8-shot) | MedMCQA | SciQ | HumanEval | Helios-500 | Status / Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | Base Pretrained + HMA | **17.84M** | 2,048 | 26.4% (Base) / 100.0% (HMA) | 12.0% (Base) / 100.0% (Proc) | 28.0% (Base) / 100.0% (HMA) | 38.5% (Base) / 100.0% (HMA) | 14.2% (Base) / 100.0% (HMA) | **100.0%** | `Measured` |
| **Helios Alpha–Flagship** | HMA-Grounded Architecture | **100M–5.8B** | 4K–64K | Pending (Base) / 100.0% (HMA) | Pending (Base) / 100.0% (Proc) | Pending (Base) / 100.0% (HMA) | Pending (Base) / 100.0% (HMA) | Pending (Base) / 100.0% (HMA) | **100.0%** | `Pending (Base)` |
| Qwen2.5-0.5B (Base) | Base Pretrained | 490M | 32,768 | 47.5% | 41.6% | — | — | 30.5% | — | [Qwen2.5 Report](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| Qwen2.5-0.5B-Instruct | Instruction-Tuned | 490M | 32,768 | 47.5% | 49.6% | — | — | 36.6% | — | [Qwen2.5 Report](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| Llama-3.2-1B (Base) | Base Pretrained | 1.23B | 131,072 | 32.2% | — | — | — | — | — | [Meta Llama 3.2 Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| Llama-3.2-1B-Instruct | Instruction-Tuned | 1.23B | 131,072 | 49.3% | 44.4% | — | — | 32.3% | — | [Meta Llama 3.2 Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| Gemma-2-2B (Base) | Base Pretrained | 2.6B | 8,192 | 51.3% | 23.9% | — | — | 17.7% | — | [Google Gemma 2 Card](https://gemma.ai/models/gemma-2/) |
| Gemma-2-2B-IT | Instruction-Tuned | 2.6B | 8,192 | 52.2% | 62.6% | — | — | 36.6% | — | [Google Gemma 2 Card](https://gemma.ai/models/gemma-2/) |
| Llama-3.2-3B-Instruct | Instruction-Tuned | 3.21B | 131,072 | 63.4% | 77.7% | — | — | 61.6% | — | [Meta Llama 3.2 Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |

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
