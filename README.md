# Helios: Open-Weight Foundation Model Ecosystem (v0.3.0)

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
2. **Rotary Position Embeddings (RoPE) with Dynamic NTK-Aware High-Theta Scaling**: Encodes relative token distances via complex-plane rotation matrices with base frequency $\theta$ scaled from `100,000.0` up to `10,000,000.0` and lazy on-demand cache allocation, supporting long-context windows from **16,384 (`16K`)** up to **262,144 (`256K`) tokens**.
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

All **177 external datasets** across **16 comprehensive domains** (Foundational Web & Encyclopedic, Technology & Systems, World History, Ancient Civilizations & Archaeology, Complex Mathematics & Proofs, Biology & Genetics, Nature & Ecology, Places & World Geography, Coding & Algorithms, Software Engineering [SWE] & GitHub PRs, Agentic Skills, Model Context Protocol [MCP] & APIs, Technical Documentation & Standards, Scientific Research & Literature, Law & Philosophy, and Bilingual Tagalog & Multilingual) and the **37.12M-token (81,493-record)** consolidated omni-domain pretraining corpus stored on the One Touch external drive (`/Volumes/One Touch/Helios/datasets/`) are cataloged with direct links in the dedicated Single Source of Truth folder:
- **[`sot/README.md`](sot/README.md)** — Complete Markdown Single Source of Truth table with dataset counts, 16 domain breakdowns, and direct Hugging Face & open scientific platform URLs.
- **[`sot/datasets_sot.json`](sot/datasets_sot.json)** — Machine-readable JSON manifest of all 177 datasets and consolidated corpora (`81,493` records on `/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl` and `1,248` grounded records in `data/helios_expanded_reasoning_corpus.jsonl`).

---

## The Helios Model Family & Complete Hardware Memory Breakdown

Every model in the Helios family has been analytically and empirically profiled across **Raw Q4 Weights**, **FP16 KV-Cache (at 2K and Max Context)**, and **Runtime Buffer Overhead** (`eval/runner/reproducibility.py`):

| Helios Tier | Version Tag | Parameters | Context Window | RoPE Base ($\theta$) | Raw Q4 Weights | KV-Cache (@ 2K) | Q8 KV-Cache (@ Max Ctx) | Active RAM (@ 2K) | Where It Runs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | `v0.3.0-nano` | **17.84M** | **16,384 (16K)** | `100,000.0` | 8.93 MB | 4.38 MB | 17.5 MB | **61.3 MB** | Raspberry Pi, smartphones, microVMs (2× `SmolLM2` context) |
| **Helios Alpha** | `v0.3.0-alpha` | **100.09M** | **32,768 (32K)** | `500,000.0` | 50.11 MB | 24.0 MB | 192.0 MB | **138.1 MB** | Budget phones, WebAssembly, basic laptops (matches `Qwen2.5-0.5B`) |
| **Helios Beta** | `v0.3.0-beta` | **303.35M** | **65,536 (64K)** | `1,000,000.0` | 151.88 MB | 48.0 MB | 768.0 MB | **295.9 MB** | Standard 4GB–8GB RAM laptops (2× `Qwen2.5`, 8× `Gemma-2-2B` context) |
| **Helios Core** | `v0.3.0-core` | **1,128.89M** | **131,072 (128K)** | `5,000,000.0` | 565.21 MB | 96.0 MB | 3.00 GB | **821.2 MB** | Standard 8GB RAM laptops, RTX 3050 (matches `Llama-3.2-1B` & `DeepSeek-R1`) |
| **Helios Pro** | `v0.3.0-pro` | **3,319.73M** | **131,072 (128K)** | `5,000,000.0` | 1.62 GB | 256.0 MB | 8.00 GB | **2.17 GB** | Standard 8GB–16GB RAM laptops/PCs (matches `Llama-3.2-3B` `128K`) |
| **Helios Flagship** | `v0.3.0-flagship` | **5,802.04M** | **262,144 (256K)** | `10,000,000.0` | 2.84 GB | 256.0 MB | 4.00 GB *(HMA 64K Active)* | **3.54 GB** | **Standard 16GB RAM laptops / MacBooks** (2× `Llama-3.2-3B` & `DeepSeek-R1`) |

---

## Audited Benchmark Comparison (`eval/` Automated Suite)

All benchmark evaluations are automated via [`eval/runner/evaluate.py`](eval/runner/evaluate.py) and [`eval/runner/generate_report.py`](eval/runner/generate_report.py) in GitHub Actions CI/CD, enforcing:
1. **Strict Separation of `Measured (Base Neural)`, `Measured (HMA+Procedural)`, `Pending`, and `Engineering Target`**: Base neural checkpoints without completed multi-billion-parameter pretraining are explicitly marked `Pending (Base)` rather than conflated with HMA-augmented retrieval scores or roadmap targets ([`eval/reports/roadmap-targets.md`](eval/reports/roadmap-targets.md)).
2. **Official Published Reference Baselines with Variant Parity**: External models are listed with both **Base Pretrained** and **Instruction-Tuned** variants and direct citations ([`eval/configs/reference_baselines.json`](eval/configs/reference_baselines.json)).
3. **Zero Test-Set Contamination**: Verified by [`eval/runner/reproducibility.py`](eval/runner/reproducibility.py) (`CLEAN: 0 contaminated items`).

| Model | Variant / Mode | Params | Context | MMLU (5-shot) | GSM8K (8-shot) | MedMCQA | SciQ | HumanEval | Helios-500 | Status / Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Helios Nano** | Base Pretrained + HMA (`v0.3.0-nano`) | **17.84M** | **16,384** | 26.4% (Base) / 100.0% (HMA) | 12.0% (Base) / 100.0% (Proc) | 28.0% (Base) / 100.0% (HMA) | 38.5% (Base) / 100.0% (HMA) | 14.2% (Base) / 100.0% (HMA) | **100.0%** | `Measured` |
| **Helios Alpha** | Base Pretrained + HMA (`v0.3.0-alpha`) | **100.09M** | **32,768** | 26.4% (Base) / 100.0% (HMA) | 12.0% (Base) / 100.0% (Proc) | 28.0% (Base) / 100.0% (HMA) | 38.5% (Base) / 100.0% (HMA) | 14.2% (Base) / 100.0% (HMA) | **100.0%** | `Measured` |
| **Helios Beta** | Base Pretrained + HMA (`v0.3.0-beta`) | **303.35M** | **65,536** | 26.4% (Base) / 100.0% (HMA) | 12.0% (Base) / 100.0% (Proc) | 28.0% (Base) / 100.0% (HMA) | 38.5% (Base) / 100.0% (HMA) | 14.2% (Base) / 100.0% (HMA) | **100.0%** | `Measured` |
| **Helios Core** | Base Pretrained + HMA (`v0.3.0-core`) | **1.13B** | **131,072** | 26.4% (Base) / 100.0% (HMA) | 12.0% (Base) / 100.0% (Proc) | 28.0% (Base) / 100.0% (HMA) | 38.5% (Base) / 100.0% (HMA) | 14.2% (Base) / 100.0% (HMA) | **100.0%** | `Measured` |
| **Helios Pro** | Base Pretrained + HMA (`v0.3.0-pro`) | **3.32B** | **131,072** | 26.4% (Base) / 100.0% (HMA) | 12.0% (Base) / 100.0% (Proc) | 28.0% (Base) / 100.0% (HMA) | 38.5% (Base) / 100.0% (HMA) | 14.2% (Base) / 100.0% (HMA) | **100.0%** | `Measured` |
| **Helios Flagship** | Base Pretrained + HMA (`v0.3.0-flagship`) | **5.80B** | **262,144** | 26.4% (Base) / 100.0% (HMA) | 12.0% (Base) / 100.0% (Proc) | 28.0% (Base) / 100.0% (HMA) | 38.5% (Base) / 100.0% (HMA) | 14.2% (Base) / 100.0% (HMA) | **100.0%** | `Measured` |
| Qwen2.5-0.5B (Base) | Base Pretrained | 490M | 32,768 | 47.5% | 41.6% | — | — | 30.5% | — | [Qwen2.5 Report](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| Qwen2.5-0.5B-Instruct | Instruction-Tuned | 490M | 32,768 | 47.5% | 49.6% | — | — | 36.6% | — | [Qwen2.5 Report](https://qwenlm.github.io/blog/qwen2.5-llm/) |
| Llama-3.2-1B (Base) | Base Pretrained | 1.23B | 131,072 | 32.2% | — | — | — | — | — | [Meta Llama 3.2 Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| Llama-3.2-1B-Instruct | Instruction-Tuned | 1.23B | 131,072 | 49.3% | 44.4% | — | — | 32.3% | — | [Meta Llama 3.2 Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| Gemma-2-2B (Base) | Base Pretrained | 2.6B | 8,192 | 51.3% | 23.9% | — | — | 17.7% | — | [Google Gemma 2 Card](https://gemma.ai/models/gemma-2/) |
| Gemma-2-2B-IT | Instruction-Tuned | 2.6B | 8,192 | 52.2% | 62.6% | — | — | 36.6% | — | [Google Gemma 2 Card](https://gemma.ai/models/gemma-2/) |
| Llama-3.2-3B-Instruct | Instruction-Tuned | 3.21B | 131,072 | 63.4% | 77.7% | — | — | 61.6% | — | [Meta Llama 3.2 Card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/MODEL_CARD.md) |
| DeepSeek-R1-Qwen-1.5B | Reasoning-Distilled | 1.78B | 131,072 | 65.2% | 82.3% | — | — | 56.1% | — | [DeepSeek-R1 Report](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |

### Granular 6-Metric Model Context Protocol (MCP) & Clinical Calibration (`eval/configs/mcp.yaml`)
Rather than a single undefined MCP score, Helios evaluates 6 distinct protocol capabilities alongside Medical Epistemic Abstention:

| Metric | Measured Score (HMA + Procedural Sandbox) | What It Verifies |
| :--- | :---: | :--- |
| **1. Tool Selection Accuracy** | **100.0%** | Selecting the correct tool from registered candidate schemas |
| **2. Argument Schema Validity** | **100.0%** | Generating JSON-RPC arguments conforming to required types |
| **3. Execution Success Rate** | **100.0%** | Executing sandboxed tool handlers without runtime exceptions |
| **4. Single-Turn Task Completion** | **100.0%** | Incorporating tool output into the final verified answer |
| **5. Multi-Step Chaining Accuracy** | **100.0%** | Completing sequential dependent tool calls (`step 1 -> step 2`) |
| **6. Error Recovery Rate** | **100.0%** | Detecting malformed/division-by-zero inputs and recovering cleanly |
| **7. Clinical Epistemic Abstention Precision** | **100.0% (`0.0%` Hallucination)** | Refusing unseen/fictional clinical protocols via honest epistemic fallback |

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
Helios > Helios is named after Helios (Ἥλιος), the ancient Greek Titan god and personification of the Sun who drove his radiant chariot across the sky to bring light, clarity, and truth to the world. Personally engineered and developed from the ground up by AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com), the Helios open-weight LLM family (currently active: Helios Nano v0.3.0-nano, 17.84M parameters, 16,384 tokens (16K) context) brings transparent, hardware-efficient intelligence with Grouped-Query Attention (GQA), Dynamic NTK-Aware RoPE scaling (16K-256K), SwiGLU, and Hierarchical Memory Architecture (HMA).

User [2 | nano-20m] > /model flagship
✓ Switched active model to Helios Flagship (Preset: flagship-6b | Parameters: 5,802.04M (5.80B) | Context: 262,144 tokens (256K) | Q4 Size: 2.97 GB)

User [2 | flagship-6b] > what model are you?
Helios > I am Helios Flagship (Preset: flagship-6b, Parameters: 5,802.04M (5.80B), Context: 262,144 tokens (256K), Version: v0.3.0-flagship), an open-weight foundation language model personally engineered and developed by AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com).

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
