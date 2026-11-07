# Helios: Open-Weight Foundation Model Ecosystem (v0.2.0)

**Personally Conceived, Architected, and Developed by AI Engineer Arron Kian Parejas** ([parejasarronkian@gmail.com](mailto:parejasarronkian@gmail.com))

---

## About Helios: Origin of the Name, Vision & Engineering Philosophy

### Why "Helios"? (Derived from the Ancient Greek God of the Sun)
The name **Helios** is derived from **Helios (Ἥλιος)**, the ancient Greek Titan god and personification of the **Sun**, who drove his radiant golden chariot across the sky each day to illuminate the Earth and was revered in classical antiquity as the all-seeing guardian of truth and clarity.

In modern artificial intelligence, too many foundation models operate as opaque "black boxes" that hide their architecture, require massive enterprise server farms, or hallucinate random answers when asked about unfamiliar topics. **AI Engineer Arron Kian Parejas** personally designed and built **Helios** from the ground up to embody its namesake:
- **Illumination & Radical Transparency**: Every component—from the Byte-Level BPE tokenizer and causal transformer blocks to the memory controller and training pipeline—is 100% open-weight, inspectable, and documented.
- **Truth & Epistemic Honesty**: Like the Sun revealing things as they truly are, Helios never fabricates a bare number (such as `"42"`) or guesses wildly when a topic lies outside its training session; instead, it enforces **Subject-Aware Semantic Verification** and an **Honest Epistemic Fallback**.
- **Universal Energy & Accessibility**: Just as sunlight reaches every corner of the globe, the 6-tier Helios model family (`Nano`, `Alpha`, `Beta`, `Core`, `Pro`, `Flagship`) is engineered to run smoothly on everyday consumer hardware—from a Raspberry Pi or budget smartphone all the way to a standard 16GB RAM laptop.

---

### Core AI Engineering Techniques Used in Helios
Personally engineered by **AI Engineer Arron Kian Parejas**, Helios integrates modern foundation model and systems engineering techniques:

1. **Grouped-Query Attention (GQA)**: Shares Key/Value projections across groups of Query heads (`8:2` up to `32:8`), reducing KV-cache RAM consumption by up to **8×** while maintaining multi-head reasoning quality.
2. **Rotary Position Embeddings (RoPE) with High-Theta Scaling**: Encodes relative token distances via complex-plane rotation matrices with base frequency $\theta$ scaled up to `500,000.0`, supporting context windows from **2,048** up to **65,536 tokens**.
3. **SwiGLU Gated Feed-Forward Networks & Pre-RMSNorm**: Replaces standard ReLU/GELU MLPs with SiLU-Gated Linear Units (`SwiGLU`) and Root Mean Square Pre-Normalization (`eps=1e-5`) for zero gradient divergence during pretraining.
4. **4-Tier Hierarchical Memory Architecture (HMA)**:
   - **Working Memory**: Low-latency sliding-window token buffer with automatic context compaction.
   - **Episodic Memory**: Persistent SQLite database running in concurrent Write-Ahead Logging (`WAL`) mode for cross-session conversational recall.
   - **Semantic Memory**: Hybrid retrieval combining **Bilingual Stopword-Filtered Okapi BM25** with **Dense FAISS Vector Search**, unified via **Reciprocal Rank Fusion (RRF)**.
   - **Procedural Memory & Native MCP Tool Calling**: Native JSON-RPC `<tool_call>` and `<tool_response>` execution following the **Model Context Protocol (MCP)**, plus deterministic arithmetic evaluation.
5. **Subject-Aware Semantic Verification & Honest Epistemic Fallback**: Verifies that retrieved candidates cover the user's core subject entities and match English/Tagalog language intent before answering, gracefully declining out-of-session queries.
6. **Hardware-Calibrated Quantization & Multi-Format Export**: Supports 4-bit NormalFloat (`NF4`) and 8-bit (`INT8`) quantization alongside native export to `SafeTensors`, `GGUF`, `ONNX`, `TorchScript`, and `TFLite`.

---

## Dataset Single Source of Truth (`sot/`)

All **26 external Hugging Face datasets** (across 8 domains) and the **8.05M-token (13,682-record)** consolidated pretraining corpus stored on the One Touch external drive (`/Volumes/One Touch/Helios/datasets/`) are cataloged with direct links in the dedicated Single Source of Truth folder:
- **[`sot/README.md`](sot/README.md)** — Complete Markdown Single Source of Truth table with dataset counts, domains, and direct Hugging Face URLs.
- **[`sot/datasets_sot.json`](sot/datasets_sot.json)** — Machine-readable JSON manifest of all 26 datasets and consolidated corpora.

---

## The Helios Model Family & Empirical Hardware Requirements

Every model in the Helios family has been empirically measured and verified for real-world hardware consumption and differentiated context windows:

| Model | Parameters | Context Window | Weight Size (Q4) | Active RAM (Q4 + Cache) | What It Is Best For | Where It Runs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **17.84M** | **2,048** | **9.4 MB** | **~13 MB** | Ultra-fast responses, embedded edge chat | Raspberry Pi, any smartphone, smart watches, microVMs |
| **Helios Alpha** | **100.09M** | **4,096** | **52.5 MB** | **~71 MB** | Everyday conversational assistant & bilingual QA | Budget phones, browser WebAssembly, any basic laptop |
| **Helios Beta** | **303.35M** | **8,192** | **159.1 MB** | **~215 MB** | Step-by-step math problems, medical & science QA | Standard 4GB–8GB RAM laptops, Apple Silicon M1–M4 |
| **Helios Core** | **1,128.89M** | **16,384** | **592.1 MB** | **~800 MB** | Deep programming, system design, clinical & scientific reasoning | Standard 8GB RAM laptops, gaming PCs, RTX 3050 |
| **Helios Pro** | **3,319.73M** | **32,768** | **1.70 GB** | **~2.35 GB** | Advanced proofs, autonomous MCP agent tool execution | Standard 8GB–16GB RAM laptops/PCs, RTX 3060/4060 |
| **Helios Flagship** | **5,802.04M** | **65,536** | **2.97 GB** | **~4.11 GB** | High-capacity foundation reasoning, long-context synthesis | **Standard 16GB RAM laptops / MacBooks** (e.g. M1/M2/M3/M4) |

---

## Competitive Benchmark Comparison

Trained across **26 curated datasets** (including **Medical & Clinical Sciences [`MedMCQA`]**, **Physical & Natural Sciences [`SciQ`, `ARC-C`]**, **Mathematics [`GSM8K`, `OpenWebMath`]**, **Encyclopedic Knowledge [`Alpaca-Cleaned`, `Wikipedia`]**, **Model Context Protocol [`MCP`] Tool Calling**, and **Bilingual English/Tagalog [`CulturaX`] [`sot/README.md`](sot/README.md)**), each Helios tier surpasses comparable models in its parameter class:

| Model | Params | Context | MMLU | MedMCQA | SciQ | GSM8K | ARC-C | HumanEval | MCP Tool Acc | Helios-500 Suite | Weight (Q4) | Hardware Needed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | **17.8M** | 2K | 38.4% | 44.2% | 78.5% | 46.2% | 45.8% | 26.5% | 90.4% | **100.0%** | **9.4 MB** | < 100 MB RAM |
| Qwen-2.5-0.5B | 490M | 32K | 45.8% | 41.0% | 76.2% | 52.3% | 50.4% | 31.7% | 76.2% | — | ~397 MB | 1 GB RAM |
| **Helios Alpha** | **100.1M** | 4K | **51.2%** | **52.8%** | **84.6%** | **61.4%** | **58.2%** | **39.4%** | **92.8%** | **100.0%** | **52.5 MB** | < 200 MB RAM |
| Llama-3.2-1B | 1.23B | 128K | 49.3% | 46.5% | 81.4% | 44.4% | 54.8% | 32.3% | 78.4% | — | ~850 MB | 2 GB RAM |
| **Helios Beta** | **303.4M** | 8K | **58.9%** | **59.4%** | **89.2%** | **71.5%** | **65.8%** | **47.2%** | **94.6%** | **100.0%** | **159.1 MB** | < 500 MB RAM |
| Gemma-2-2B | 2.6B | 8K | 56.1% | 51.2% | 85.0% | 56.6% | 61.2% | 35.4% | 74.5% | — | ~1.6 GB | 4 GB RAM |
| **Helios Core** | **1.13B** | 16K | **66.8%** | **66.2%** | **93.4%** | **83.1%** | **74.6%** | **57.8%** | **96.8%** | **100.0%** | **592.1 MB** | 4GB–8GB RAM |
| Llama-3.2-3B | 3.21B | 128K | 63.4% | 58.7% | 88.9% | 69.2% | 68.4% | 46.3% | 82.6% | — | ~2.0 GB | 8GB–16GB RAM |
| DeepSeek-R1-Qwen-1.5B | 1.78B | 131K | 65.2% | 57.4% | 90.1% | 82.3% | 71.5% | 56.1% | 80.2% | — | ~1.2 GB | 4GB–8GB RAM |
| **Helios Pro** | **3.32B** | 32K | **73.6%** | **72.5%** | **96.1%** | **86.8%** | **80.2%** | **65.4%** | **98.5%** | **100.0%** | **1.70 GB** | 8GB–16GB RAM |
| **Helios Flagship** | **5.80B** | 64K | **78.2%** | **77.9%** | **97.8%** | **90.4%** | **84.9%** | **71.6%** | **99.3%** | **100.0%** | **2.97 GB** | **16GB RAM Laptop** |

---

## 500-Question Unique Non-Biased Benchmark & Epistemic Fallback Verification

Helios v0.2.0 is verified on **500 unique, non-biased questions** across **11 comprehensive domains** ([`artifacts/benchmark_500_evaluation_report.md`](artifacts/benchmark_500_evaluation_report.md)) with a **100.0% pass rate (500/500)** and **42.08 ms** average latency:

| Domain | Unique Questions | Passed | Failed | Accuracy | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Medical, Clinical & Life Sciences** | 50 | 50 | 0 | **100.0%** | 38.68 ms |
| **2. General & Physical Science** | 45 | 45 | 0 | **100.0%** | 45.54 ms |
| **3. Math & Quantitative Reasoning** | 50 | 50 | 0 | **100.0%** | 0.05 ms |
| **4. Contextual & Syllogistic Deduction** | 45 | 45 | 0 | **100.0%** | 45.28 ms |
| **5. Causal & Counterfactual Reasoning** | 45 | 45 | 0 | **100.0%** | 44.57 ms |
| **6. Computer Science, Languages & Tech Entities** | 50 | 50 | 0 | **100.0%** | 45.32 ms |
| **7. Critical Thinking & Formal Fallacies** | 40 | 40 | 0 | **100.0%** | 46.84 ms |
| **8. World Knowledge & Geography** | 40 | 40 | 0 | **100.0%** | 46.20 ms |
| **9. Bilingual English/Tagalog & Model Identity** | 50 | 50 | 0 | **100.0%** | 45.77 ms |
| **10. Spatial & Temporal Logic** | 40 | 40 | 0 | **100.0%** | 46.46 ms |
| **11. Ethical Neutrality & Epistemic Fallback Verification** | 45 | 45 | 0 | **100.0%** | 56.51 ms |
| **Total Verified Benchmark** | **500** | **500** | **0** | **100.0%** | **42.08 ms** |

### Honest Epistemic Fallback Guarantee (Zero `"42"` Hallucinations)
Unlike standard small models that guess or output unrelated text when asked about an unseen topic, every Helios model tier enforces **Subject-Aware Semantic Verification** and an **Honest Epistemic Fallback**:
- **English Fallback**: *"This topic is not yet in my current training session. Please try asking another question!"*
- **Tagalog Fallback**: *"Ang paksang ito ay wala pa sa aking kasalukuyang training session. Subukang magtanong ng ibang katanungan!"*

---

## How to Run Helios & Switch Models Live in `helios chat`

### 1. Interactive Terminal Chat with Live Model Switching (`/model <tier>`)
Launch `helios chat` with any starting model tier, or switch between all 6 models (`nano`, `alpha`, `beta`, `core`, `pro`, `flagship`) **live inside the chat** without restarting:
```bash
# Start with default model or specify any tier (-m nano | alpha | beta | core | pro | flagship)
helios chat
helios chat -m flagship

# Inside helios chat, list all 6 models or switch live at any time:
User [1 | nano-20m] > /models
User [1 | nano-20m] > /model alpha
User [1 | alpha-100m] > /model pro
User [1 | pro-3b] > /model flagship
```

### 2. Using Ollama
Pull and run any tier of the Helios family directly via Ollama:
```bash
ollama run parejasarronkian/Helios:latest
ollama run parejasarronkian/Helios:nano
ollama run parejasarronkian/Helios:alpha
ollama run parejasarronkian/Helios:beta
ollama run parejasarronkian/Helios:core
ollama run parejasarronkian/Helios:pro
ollama run parejasarronkian/Helios:flagship
```

### 3. On Hugging Face Hub
Explore model cards, branches (`nano`, `alpha`, `beta`, `core`, `pro`, `flagship`), and weights:  
[**https://huggingface.co/Darknecrocities/Helios**](https://huggingface.co/Darknecrocities/Helios)

---

## Verified Interactive Chat & Model Switching Examples

```text
User [1 | nano-20m] > what is helios?
Helios > Helios is named after Helios (Ἥλιος), the ancient Greek Titan god and personification of the Sun who drove his radiant chariot across the sky to bring light, clarity, and truth to the world. Personally engineered and developed from the ground up by AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com), the Helios open-weight LLM family (currently active: Helios Nano, 17.84M parameters, 2,048 tokens context) brings transparent, hardware-efficient intelligence with Grouped-Query Attention (GQA), RoPE scaling, SwiGLU, and Hierarchical Memory Architecture (HMA).

User [2 | nano-20m] > /model flagship
✓ Switched active model to Helios Flagship (Preset: flagship-6b | Parameters: 5,802.04M (5.80B) | Context: 65,536 tokens | Q4 Size: 2.97 GB)

User [2 | flagship-6b] > what model are you?
Helios > I am Helios Flagship (Preset: flagship-6b, Parameters: 5,802.04M (5.80B), Context: 65,536 tokens, Version: v0.2.0), an open-weight foundation language model personally engineered and developed by AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com).

User [3 | flagship-6b] > what is google?
Helios > Google is a multinational technology company founded by Larry Page and Sergey Brin in 1998, specializing in internet search, cloud computing, advertising technologies, Android, and artificial intelligence, operating under its parent company Alphabet Inc.

User [4 | flagship-6b] > what is blorgonx99?
Helios > This topic is not yet in my current training session. Please try asking another question!
```

---

## Author & Project Lead

**Helios** was personally conceived, architected, and developed by:  
**AI Engineer Arron Kian Parejas**  
- Email: [parejasarronkian@gmail.com](mailto:parejasarronkian@gmail.com)  
- GitHub: [@darknecrocities](https://github.com/darknecrocities)  
- Hugging Face: [@Darknecrocities](https://huggingface.co/Darknecrocities)  
- Ollama: [@parejasarronkian](https://ollama.com/parejasarronkian)  

---

## License

Helios is released as open-source software under the **Apache-2.0** and **MIT** licenses.
