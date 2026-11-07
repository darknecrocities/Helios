# Helios Open Internet Pretraining Datasets & Token Allocation Specification

**Architect & Author**: Arron Kian Parejas (`arronkianparejas@gmail.com`)  
**Repository**: `https://github.com/darknecrocities/Helios`  
**License**: Apache-2.0 / Open Data Permissive  

---

## 1. Executive Summary & Pretraining Philosophy

To initiate immediate pretraining across all Helios model stages (Nano 15M–35M, Alpha 100M, Beta 300M, Core 1B, Pro 3B, Flagship 6B), this document specifies an open, high-quality, reproducible dataset mixture sourced from publicly available internet repositories on Hugging Face.

Helios adopts an **aggressive overtraining strategy** beyond compute-optimal Chinchilla ratios ($D \approx 20N$ tokens):
- Chinchilla compute-optimal training prioritizes minimum compute during pretraining.
- In contrast, Helios prioritizes **inference efficiency, local responsiveness on edge devices (such as Apple Silicon M-series laptops), and superior multilingual/code representation**.
- Small models (Nano, Alpha, Beta) are trained on $85\times$ to $200\times$ tokens per parameter, packing maximum reasoning density into compact hidden dimensions.

---

## 2. Curated Hugging Face Dataset Catalog (12 Datasets)

| # | Domain | Dataset Name | Hugging Face Dataset ID | License | Available Tokens | Primary Strength |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **English Educational** | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | 1.3T / 5.4T | High educational density via Llama-3 scoring |
| 2 | **English General** | FineWeb | `HuggingFaceFW/fineweb` | ODC-By 1.0 | 15.0T | 10+ heuristic filters, MinHash deduplicated |
| 3 | **English Benchmark** | DCLM-Baseline | `mlfoundations/dCLM-baseline` | Apple / ODC-By | 3.8T | FastText filtered, state-of-the-art open baseline |
| 4 | **Synthetic Textbooks** | Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | Apache-2.0 | ~30M docs (~35B tok) | Mixtral-8x22B synthetic textbooks and tutorials |
| 5 | **STEM & Mathematics** | OpenWebMath | `open-web-math/open-web-math` | ODC-By 1.0 | 14.7B | Clean LaTeX math equations & forum discussions |
| 6 | **Formal Reasoning** | Proof-Pile-2 | `EleutherAI/proof-pile-2` | Apache-2.0 / Open | 55.0B | arXiv papers, PubMed literature, formal code |
| 7 | **Permissive Code** | The Stack v2 | `bigcode/the-stack-v2` | BigCode OpenRAIL-M | 3B+ files | 600+ languages, strict license filtering |
| 8 | **Permissive Code** | StarCoder Data | `bigcode/starcoderdata` | BigCode OpenRAIL-M | 250.0B | Permissively licensed GitHub code & commits |
| 9 | **Filipino Cleaned** | CulturaX (tl) | `uonlp/CulturaX` (subset `tl`) | ODC-By 1.0 | 1.2B | Deduplicated mC4 + OSCAR Tagalog split |
| 10 | **Filipino Web Crawl** | OSCAR-2301 (tl) | `oscar-corpus/OSCAR-2301` (subset `tl`) | CC0 / Open | 1.5B | Filtered Common Crawl Tagalog pages |
| 11 | **Filipino Monolingual** | CC-100 (tl) | `statmt/cc-100` (subset `tl`) | CC-BY-NC 4.0 | 600M | FastText filtered Tagalog corpus |
| 12 | **Filipino Knowledge** | Wikipedia (tl) | `wikimedia/wikipedia` (`20231101.tl`) | CC-BY-SA 4.0 | 25M | High-precision Tagalog encyclopedia articles |

---

## 3. Global Domain Mixture Ratio

Across all Helios pretraining stages, data is sampled according to the following balanced ratio:

```
+------------------------------------------------------------------------+
|                      HELIOS PRETRAINING MIXTURE                        |
+------------------------------------------------------------------------+
|  [30%] FineWeb-Edu          - High-density educational & encyclopedic  |
|  [15%] FineWeb / DCLM       - Broad web crawl, real-world knowledge    |
|  [15%] Cosmopedia v2        - Synthetic textbook & concept narratives  |
|  [20%] The Stack / StarCoder- Permissive source code & documentation   |
|  [10%] OpenWebMath / Proof  - Formal math, scientific papers, LaTeX    |
|  [10%] Filipino / Multilingual - CulturaX (tl), OSCAR (tl), Wiki (tl)  |
+------------------------------------------------------------------------+
```

### Domain Allocation Rationale:
1. **30% FineWeb-Edu + 15% Cosmopedia v2 (45% high-density learning)**: Crucial for small models (15M–300M). High-quality synthetic and educational tokens prevent learning decay and accelerate zero-shot generalization.
2. **20% Code (The Stack v2 + StarCoder)**: Essential for structuring logical step-by-step thinking, instruction adherence, and tool/API invocation (HMA procedural memory).
3. **10% Math & Science (OpenWebMath + Proof-Pile-2)**: Preserves numerical precision and prevents hallucination in quantitative contexts.
4. **10% Filipino (CulturaX, OSCAR, CC-100, Wiki)**: Fulfills the foundational mandate for strong English/Filipino bilingual proficiency, matching the fertility optimizations in `helios.tokenizer`.

---

## 4. Parameter Scaling Roadmap: Token Allocations Across All 6 Model Tiers

| Stage | Model Preset | Target Parameters | Chinchilla Baseline | Helios Pretraining Tokens | Overtraining Ratio | Recommended Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nano** | `nano-15m` / `nano-35m` | 15M–35M | 0.3B–0.7B | **3.0 Billion** | 85x–200x | Apple Silicon M3 (16GB) / 1x RTX 4090 |
| **Alpha** | `alpha-100m` | 100M | 2.0B | **15.0 Billion** | 150x | 1x–2x A100 (80GB) |
| **Beta** | `beta-300m` | 300M | 6.0B | **45.0 Billion** | 150x | 4x A100 / H100 |
| **Core** | `core-1b` | 1.0B | 20.0B | **150.0 Billion** | 150x | 8x H100 Node |
| **Pro** | `pro-3b` | 3.0B | 60.0B | **450.0 Billion** | 150x | 4x–8x H100 Nodes (32 GPUs) |
| **Flagship**| `flagship-6b` | 6.0B | 120.0B | **1.2 Trillion** | 200x | Distributed Slurm/Kubernetes Cluster |

---

## 5. Exact Token Counts by Sub-Corpus per Model Stage

### Stage 1: Helios Nano (15M–35M) — 3.0 Billion Tokens
- FineWeb-Edu: **900 Million tokens**
- FineWeb / DCLM: **450 Million tokens**
- Cosmopedia v2: **450 Million tokens**
- The Stack v2 / StarCoder (Python, JS, Rust, SQL): **600 Million tokens**
- OpenWebMath & Proof-Pile-2: **300 Million tokens**
- Filipino (CulturaX `tl` 250M + Wikipedia `tl` 50M): **300 Million tokens**

### Stage 2: Helios Alpha (100M) — 15.0 Billion Tokens
- FineWeb-Edu: **4.50 Billion tokens**
- FineWeb / DCLM: **2.25 Billion tokens**
- Cosmopedia v2: **2.25 Billion tokens**
- Code (Top 15 languages): **3.00 Billion tokens**
- Math & Science: **1.50 Billion tokens**
- Filipino (CulturaX `tl` + OSCAR `tl`): **1.50 Billion tokens**

### Stage 3: Helios Beta (300M) — 45.0 Billion Tokens
- FineWeb-Edu: **13.50 Billion tokens**
- FineWeb / DCLM: **6.75 Billion tokens**
- Cosmopedia v2: **6.75 Billion tokens**
- Code: **9.00 Billion tokens**
- Math & Science: **4.50 Billion tokens**
- Filipino & Multilingual: **4.50 Billion tokens**

### Stage 4: Helios Core (1B) — 150.0 Billion Tokens
- FineWeb-Edu: **45.0 Billion tokens**
- FineWeb / DCLM: **22.5 Billion tokens**
- Cosmopedia v2: **22.5 Billion tokens**
- Code: **30.0 Billion tokens**
- Math & Science: **15.0 Billion tokens**
- Filipino & Multilingual: **15.0 Billion tokens**

### Stage 5: Helios Pro (3B) — 450.0 Billion Tokens
- FineWeb-Edu: **135.0 Billion tokens**
- FineWeb / DCLM: **67.5 Billion tokens**
- Cosmopedia v2: **67.5 Billion tokens**
- Code: **90.0 Billion tokens**
- Math & Science: **45.0 Billion tokens**
- Filipino & Multilingual: **45.0 Billion tokens**

### Stage 6: Helios Flagship (6B) — 1.2 Trillion Tokens
- FineWeb-Edu: **360.0 Billion tokens**
- FineWeb / DCLM: **180.0 Billion tokens**
- Cosmopedia v2: **180.0 Billion tokens**
- Code: **240.0 Billion tokens**
- Math & Science: **120.0 Billion tokens**
- Filipino & Multilingual: **120.0 Billion tokens**

---

## 6. Pre-tokenization & Sharding Recipe

Data is pre-tokenized offline using `helios.tokenizer.HeliosTokenizer` (Byte-level BPE, 32,000 vocab) and stored as contiguous memory-mapped binary arrays (`.bin` / uint16 tokens with `.idx` metadata):
1. **Chunking**: Sequences are packed into fixed context windows ($L = 2,048$) separated by `<|endoftext|>`.
2. **Sharding**: Shards are sized at 100,000 sequences (~204.8M tokens per `.bin` file, ~410MB on disk).
3. **Cursor Integration**: `CursorDataLoader` consumes shards sequentially, recording `{shard_id, sequence_idx, token_offset}` in every checkpoint manifest.
4. **Integrity**: Each shard writes a SHA-256 hash in `dataset_manifest.json` ensuring tamper-evident provenance.
