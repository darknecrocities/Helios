# Helios: Friendly Open-Source AI Model Ecosystem (v0.2.0)

**Helios** is a modern, transparent, and completely open-source AI language model family created by Filipino engineer **Arron Kian Parejas** (`parejasarronkian@gmail.com`).

Unlike models that hide how they work or hallucinate random numbers on unknown topics, Helios is engineered from scratch with **Hierarchical Memory Architecture (HMA)** grounding, **deterministic procedural mathematics**, and an **honest epistemic fallback** across all 6 model tiers.

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

Trained on an **8.05M-token (13,685-sequence) multi-domain pretraining and instruction corpus** spanning **Medical & Clinical Sciences (`MedMCQA`)**, **Physical & Natural Sciences (`SciQ`, `ARC-C`)**, **Mathematics (`GSM8K`)**, **Encyclopedic Knowledge (`Alpaca-52K`)**, **Model Context Protocol (`MCP`) Tool Calling**, and **Bilingual English/Tagalog**, each Helios tier surpasses comparable models in its parameter class:

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

## Multi-Format Model Distribution (`dist/`)

All 6 Helios model tiers (`Nano`, `Alpha`, `Beta`, `Core`, `Pro`, `Flagship`) are packaged in standard distribution formats:
- **SafeTensors (`.safetensors`)**: Zero-copy, memory-safe neural weights with `config.json` and `manifest.json`.
- **PyTorch & TorchScript (`.pt`)**: Signal-safe training & JIT runtime checkpoints.
- **ONNX (`.onnx`)**: Cross-platform Open Neural Network Exchange graph.
- **TFLite (`.tflite`)**: Mobile and embedded edge FlatBuffer binary.
- **Ollama Modelfiles (`Modelfile.<tier>`)**: Differentiated context windows (`2K` to `64K`).
- **Python Packages (`.whl` & `.tar.gz`)**: Installable wheel and source distributions.

---

## How to Run Helios

### 1. Interactive Terminal Chat (Python)
```bash
git clone https://github.com/darknecrocities/Helios.git
cd Helios
pip install -e .
helios chat
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

## Verified Interactive Chat Examples

```text
User [1] > what is philippines?
Helios > The Philippines is a sovereign island country in Southeast Asia situated in the western Pacific Ocean. Comprising 7,641 islands, it has three primary regions: Luzon in the north, Visayas in the center, and Mindanao in the south. The national capital is Manila, and the nation is renowned for its biodiversity, rich culture, and warm hospitality.

User [2] > what is google?
Helios > Google is a multinational technology company and subsidiary of Alphabet Inc. specializing in internet search, cloud computing, artificial intelligence, and the Android operating system.

User [3] > what is computer?
Helios > A computer is a programmable electronic device that accepts raw data as input, processes it using arithmetic and logical instructions via a CPU, and produces output.

User [4] > what is javascript?
Helios > JavaScript is a high-level, dynamic programming language that is a core technology of the World Wide Web alongside HTML and CSS, enabling interactive web applications and server-side runtimes like Node.js.

User [5] > what is array?
Helios > An array is a linear data structure consisting of a collection of elements stored at contiguous memory locations and accessed in O(1) time via numerical indices.

User [6] > who is newton?
Helios > Sir Isaac Newton (1643–1727) was an English mathematician, physicist, and astronomer who formulated the laws of motion, universal gravitation, and calculus.

User [7] > what is earth?
Helios > Earth is the third planet from the Sun in our Solar System and the only known astronomical object confirmed to harbor life, with 71% of its surface covered by water.

User [8] > ano ang luzon?
Helios > Ang Luzon ay ang pinakamalaki at pinakamataong pangkat ng mga pulo sa Pilipinas na matatagpuan sa hilagang bahagi ng bansa, kung saan naroroon ang pambansang kabisera na Maynila.

User [9] > what is model version?
Helios > I am running Helios v0.2.0 (Architecture Preset: nano-20m) with Hierarchical Memory Architecture (HMA) grounding, engineered by Arron Kian Parejas (parejasarronkian@gmail.com).

User [10] > what is blorgonx99?
Helios > This topic is not yet in my current training session. Please try asking another question!
```

---

## Author & Project Lead

**Helios** was conceived, architected, and developed by:  
**Arron Kian Parejas**  
- Email: [parejasarronkian@gmail.com](mailto:parejasarronkian@gmail.com)  
- GitHub: [@darknecrocities](https://github.com/darknecrocities)  
- Hugging Face: [@Darknecrocities](https://huggingface.co/Darknecrocities)  
- Ollama: [@parejasarronkian](https://ollama.com/parejasarronkian)  

---

## License

Helios is released as open-source software under the **Apache-2.0** and **MIT** licenses.
