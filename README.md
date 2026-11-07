# Helios: Open-Weight Foundation Model Ecosystem

Helios is a modular, transparent, open-weight foundation model ecosystem designed for local and distributed execution. The project decouples parametric token generation from dynamic knowledge representation and long-term state, resolving foundational limitations in frontier models such as catastrophic forgetting, context-window degradation, and KV-cache memory bloat.

---

## Architectural Overview

Helios establishes clean boundary isolation between parametric representation, external state, and runtime tooling:

```
       +--------------------------------------------------------+
       |                       Helios CLI                       |
       +--------------------------------------------------------+
               |                    |                   |
               v                    v                   v
     +-------------------+  +---------------+  +-----------------+
     |  helios.tokenizer |  | helios.model  |  | helios.training |
     +-------------------+  +---------------+  +-----------------+
               |                    ^                   ^
               |                    |                   |
               v                    |                   |
     +--------------------------------------+           |
     |         helios.memory (HMA)          |           |
     |  Working | Episodic | Semantic | Proc|-----------+
     +--------------------------------------+
```

### Core Subsystems

1. **Decoder-Only Transformer (`helios.model`)**:
   - Modern causal language model architecture with Pre-RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU gated MLPs, and Grouped-Query Attention (GQA).
   - Dynamic KV-cache support with 3:1 query-to-KV head ratio, reducing inference memory bandwidth by 66.7%.
   - Helios Nano target sizing (10M–40M parameters) designed for empirical training and inference on edge hardware (e.g., Apple Silicon unified memory).

2. **Hierarchical Memory Architecture (`helios.memory`)**:
   - **Working Memory**: In-memory bounded task buffer with sliding context management.
   - **Episodic Memory**: Persistent SQLite store with Write-Ahead Logging (WAL) and session isolation.
   - **Semantic Memory**: Hybrid dense vector retrieval (FAISS) and sparse lexical search (BM25) synthesized via Reciprocal Rank Fusion (RRF).
   - **Procedural Memory**: Schema-verified tool and skill execution registry.
   - **Weight Isolation Barrier**: Strict physical separation preventing runtime interactions from polluting static model parameters.

3. **Training Controller with Interruption Fallback (`helios.training`)**:
   - Microbatch boundary POSIX signal trapping (`SIGINT`, `SIGTERM`) for zero-loss graceful halts.
   - Atomic two-phase checkpointing (`.pt.tmp` -> `.pt`) with SHA-256 integrity verification.
   - Exact 6-state resumption restoring weights, AdamW optimizer, LR scheduler, multi-RNG seeds, dataloader cursors, and iteration metrics.

4. **Tokenization Engine (`helios.tokenizer`)**:
   - Lossless byte-level BPE and Unigram tokenizers.
   - Dedicated special token contracts for chat headers and tool execution schemas.
   - Multilingual fertility optimization prioritizing English, Filipino, and programming languages.

---

## Model Targets

The initial architecture baseline is focused on the **Helios Nano** family:

| Preset | Target Parameters | Hidden Dimension ($d$) | Layers | Heads ($Q$) | KV Heads ($KV$) | Feed-Forward ($d_{ff}$) | Max Sequence Length |
|---|---|---|---|---|---|---|---|
| `nano-15m` | ~15M | 256 | 6 | 4 | 2 | 680 | 1,024 |
| `nano-20m` | ~20M | 288 | 6 | 6 | 2 | 768 | 1,024 |
| `nano-25m` | ~25M | 320 | 8 | 8 | 2 | 864 | 1,024 |
| `nano-35m` | ~35M | 384 | 10 | 6 | 2 | 1,024 | 1,024 |

---

## Quickstart

### Prerequisites
- Python 3.11, 3.12, or 3.13
- [uv](https://docs.astral.sh/uv/) (recommended package installer)

### Installation

```bash
# Clone the repository
git clone https://github.com/darknecrocities/Helios.git
cd Helios

# Create virtual environment with Python 3.13
uv venv .venv --python 3.13
source .venv/bin/activate

# Install in editable mode with development tools
uv pip install -e ".[dev]"

# Optionally install memory dependencies (FAISS, BM25)
uv pip install -e ".[memory]"
```

### CLI Usage

```bash
# Print ecosystem version
helios version

# Inspect model presets and target parameters
helios inspect --model-size nano-20m

# Display CLI help menu
helios --help
```

---

## Milestone Roadmap

- **Milestone 1 (Current)**: Foundation repository setup, packaging (PEP 517/621), open-source governance suite, CI workflow matrix, and frontier model problem research document.
- **Milestone 2**: Byte-level BPE and Unigram tokenizer engines, special tokens contract, bilingual fertility evaluation, and HuggingFace interoperability.
- **Milestone 3**: Decoder-only Transformer implementation (RoPE, Pre-RMSNorm, SwiGLU, GQA, KV-cache) and parameter inspector tool.
- **Milestone 4**: Hierarchical Memory Architecture (HMA) with Working, Episodic, Semantic, and Procedural tiers, plus Weight Isolation Barrier.
- **Milestone 5**: Robust training controller, POSIX signal trapping, exact 6-state resumption, and atomic checkpointing.
- **Milestone 6**: End-to-end verification across Tiers 1-4 and adversarial coverage hardening (Tier 5).

---

## License & Citation

Helios source code is dual-licensed under the **Apache License, Version 2.0** or the **MIT License**. See [LICENSE](LICENSE) for details. Pretrained model checkpoints and curated datasets maintain independent licensing and model cards.

If you use Helios in your research, please cite:

```bibtex
@software{helios_llm_2026,
  author = {Helios Open Source Contributors},
  title = {Helios: An Open-Weight Foundation Model Ecosystem with Decoupled Hierarchical Memory},
  url = {https://github.com/darknecrocities/Helios},
  version = {0.1.0},
  year = {2026}
}
```
