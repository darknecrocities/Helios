# Project: Helios Open-Weight LLM

## Architecture
Helios is a modular, transparent, open-weight foundation model ecosystem designed for local and distributed execution. The system decouples parametric representation from dynamic knowledge and long-term memory to prevent catastrophic forgetting and KV-cache bloat.

### Module Boundaries & Data Flow
1. **Core Package (`helios`)**:
   - `helios.tokenizer`: Byte-level BPE and Unigram tokenizers, special token registry, bilingual (EN/Filipino) and code fertility optimization.
   - `helios.model`: PyTorch decoder-only causal language model with Pre-RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU gated MLPs, Grouped-Query Attention (GQA), and KV caching.
   - `helios.memory`: Hierarchical Memory Architecture (HMA) with 4 distinct tiers: Working (in-memory bounded), Episodic (SQLite/WAL), Semantic (FAISS dense + BM25 sparse + Reciprocal Rank Fusion), and Procedural (executable skills/tools).
   - `helios.training`: Robust training controller with POSIX signal trapping (`SIGINT`, `SIGTERM`), atomic checkpointing, and exact 6-state resumption (weights, AdamW optimizer, scheduler, multi-RNGs, dataloader cursors, and run metadata).
   - `helios.cli`: Command-line entry points for parameter inspection, tokenizer training/encoding, memory queries, and training runs.
2. **E2E Test Suite (`tests/e2e`)**:
   - Requirement-driven, opaque-box testing harness validating CLI interfaces, serialization, and end-to-end integration across Tiers 1–4, followed by adversarial coverage hardening in Tier 5.

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

---

## Feature Inventory
Every feature from the Phase 0 Survey is inventoried and assigned to a milestone. No feature is unassigned.

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Git Remote & Branch Setup | Pair local repo with `https://github.com/darknecrocities/Helios`, branch `main`, initial commit, PR convention | M1 | ORIGINAL_REQUEST §1 |
| 2 | Python Packaging & Project Config | `pyproject.toml` with PEP 517/621, Python 3.11+ requirement, dependencies (torch, transformers, datasets, tokenizers, pytest), ruff/black linter configs | M1 | Master Plan §16 |
| 3 | Open Source Governance Files | `README.md`, `LICENSE` (Apache 2.0 / MIT dual), `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CITATION.cff` | M1 | ORIGINAL_REQUEST §7, Master Plan §14.5 |
| 4 | Research Model Problem Analysis | Comprehensive analysis of Astra, Opus, Gemini, and Muse failure modes (epistemic uncertainty, workarounds, context illusion, catastrophic forgetting) and architectural mitigations | M1 | ORIGINAL_REQUEST §2, Master Plan §2 |
| 5 | GitHub Actions CI Pipeline | Automated linting, type-checking, unit tests, and integration smoke tests on pull requests | M1 | Master Plan §16 |
| 6 | Byte-Level BPE Tokenizer Engine | Lossless UTF-8 byte-level BPE tokenizer with 32K baseline vocabulary | M2 | ORIGINAL_REQUEST §4, Master Plan §5.1 |
| 7 | Unigram Tokenizer Engine | SentencePiece-style Unigram tokenizer implementation for comparative tokenization | M2 | ORIGINAL_REQUEST §4, Master Plan §5.1 |
| 8 | Special Tokens Contract | BOS, EOS, PAD, chat role headers (`<|start_header_id|>`, `<|end_header_id|>`), tool call/return markers (`<|call:tool|>`, `<|return:tool|>`) | M2 | Master Plan §5.2 |
| 9 | Multilingual & Code Fertility Evaluation | Fertility analysis and test benchmarks across English, Filipino, and code snippets | M2 | Master Plan §5.1 |
| 10 | Tokenizer Serialization & HuggingFace Interop | Export and load from `tokenizer.json`, `tokenizer_config.json`, and `special_tokens_map.json` | M2 | Master Plan §14.2 |
| 11 | Helios Model Configuration | Strongly typed `HeliosConfig` supporting Nano-15M, Nano-25M, and Nano-35M presets with validation | M3 | Master Plan §4.1, §4.2 |
| 12 | Token Embeddings & Weight Tying | Learned embedding matrix with configurable tying to output language model head | M3 | Master Plan §4.1 |
| 13 | Rotary Position Embeddings (RoPE) | Relative position representation with precomputed frequency caches and context scaling | M3 | Master Plan §4.1 |
| 14 | Pre-RMSNorm Normalization | Numerically stable root-mean-square normalization with configurable epsilon | M3 | Master Plan §4.1 |
| 15 | Grouped-Query Attention (GQA) & KV Cache | Multi-head query with grouped key/value heads (e.g. 3:1 ratio), causal masking, and stateful dynamic `KVCache` | M3 | Master Plan §4.1 |
| 16 | SwiGLU Gated Feed-Forward | Swish-gated linear unit with expansion ratio $d_{ff} \approx \frac{8}{3} d_{model}$ rounded to multiple of 64 | M3 | Master Plan §4.1 |
| 17 | Helios Decoder-Only Transformer | Full `HeliosForCausalLM` stack combining Pre-RMSNorm, GQA, RoPE, and SwiGLU with causal cross-entropy loss | M3 | Master Plan §4.1 |
| 18 | Parameter Inspector Tool | Exact non-duplicating parameter counter `count_parameters(model)` reporting total vs trainable and layer breakdown | M3 | ORIGINAL_REQUEST §3, Master Plan §4.2 |
| 19 | Nano Target Sizing Verification | Empirical verification that Nano configuration fits strictly within 10M–40M parameters | M3 | Master Plan §4.2 |
| 20 | Working Memory Buffer | Short-lived, in-memory bounded task buffer with sliding window context management | M4 | ORIGINAL_REQUEST §5, Master Plan §10.1 |
| 21 | Episodic Memory Store | Persistent SQLite store with WAL mode, timestamped turns, session tagging, and user retention policies | M4 | Master Plan §10.1 |
| 22 | Semantic Memory (Hybrid Dense + Sparse) | Dense vector search (FAISS / cosine) combined with sparse BM25 indexing | M4 | Master Plan §10.1, §10.2 |
| 23 | Reciprocal Rank Fusion (RRF) | Multi-retrieval rank fusion combining dense and sparse scores with source trust filtering | M4 | Master Plan §10.2 |
| 24 | Procedural Memory & Skill Registry | Structured store of verified tool definitions, schemas, and execution workflows | M4 | Master Plan §10.1 |
| 25 | Memory Governance & Weight Isolation | External store interface preventing parametric weight pollution, with prompt injection sanitization | M4 | ORIGINAL_REQUEST §5, Master Plan §10.3, §10.4 |
| 26 | Training Configuration & Loss Tracking | Immutable `TrainingConfig`, causal next-token cross-entropy, and gradient accumulation control | M5 | Master Plan §7.1, §7.2 |
| 27 | L2 Gradient Clipping & NaN/Inf Abort | Gradient norm clipping with anomaly detection, NaN/Inf loss-spike abort, and rollback mechanism | M5 | Master Plan §7.2, §7.4 |
| 28 | Atomic Checkpoint Management | Atomic write-and-rename checkpointing (`.pt.tmp` -> `.pt`) with SHA-256 integrity verification | M5 | Master Plan §7.4 |
| 29 | POSIX Signal Trapping & Interruption Fallback | Trapping `SIGINT` and `SIGTERM` on microbatch boundaries for graceful shutdown and auto-save | M5 | ORIGINAL_REQUEST §6, Master Plan §7.4 |
| 30 | Exact 6-State Resumption Engine | Full checkpoint restoration: weights, AdamW optimizer, LR scheduler, multi-RNGs, dataloader cursors, and metrics | M5 | ORIGINAL_REQUEST §6, Master Plan §7.4 |
| 31 | Overfit-One-Batch Smoke Test | Rapid verification utility ensuring model and optimizer drive training loss to near-zero on a single batch | M5 | Master Plan §17 |
| 32 | Opaque-Box E2E Test Suite (Tiers 1-4) | Comprehensive requirement-driven test suite with >=11*N test cases covering all inventoried features | M6 / E2E | Dual Track Spec |
| 33 | Adversarial Coverage Hardening (Tier 5) | White-box stress-testing, boundary probing, and coverage gap closure via Challengers | M6 | Dual Track Spec |

---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Repository Foundation, Packaging & Research Analysis | Git remote pairing (`https://github.com/darknecrocities/Helios`), `pyproject.toml`, open source governance (README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION), CI pipeline, research model problems document | None | DONE |
| M2 | Tokenizer Engine (Byte-Level BPE & Unigram) | `helios/tokenizer/`: Byte-level BPE, Unigram, special tokens contract, EN/Filipino/code fertility benchmarks, HF serialization formats | M1 | DONE |
| M3 | Decoder-Only Transformer Architecture & Parameter Inspection | `helios/model/`: RoPE, Pre-RMSNorm, SwiGLU, GQA, KV Cache, `HeliosForCausalLM`, `HeliosConfig` (Nano 10M–40M), and `parameter_inspector.py` | M1 | DONE |
| M4 | Hierarchical Memory Architecture (HMA) & Hybrid Retrieval | `helios/memory/`: Working, Episodic (SQLite/WAL), Semantic (Dense/BM25 + RRF), Procedural (Skills), external store controller & weight isolation | M1, M2 | DONE |
| M5 | Training Controller & Interruption Fallback Engine | `helios/training/`: POSIX signal trapping (`SIGINT`/`SIGTERM`), atomic checkpointing, exact 6-state resumption, L2 grad clipping, overfit-one-batch smoke test | M1, M3 | DONE |
| M6 | Final Milestone: 100% E2E Test Suite Pass & Adversarial Hardening | Phase 1: Pass 100% of Tiers 1-4 tests published in `TEST_READY.md`. Phase 2: Tier 5 adversarial coverage hardening with Challengers | M1, M2, M3, M4, M5, E2E | PLANNED |

### Parallel Track: E2E Testing Track
- **Owner**: E2E Testing Track Orchestrator
- **Scope**: Requirement-driven test harness (`tests/e2e/`), test suites for Tiers 1–4, publishing `TEST_READY.md`.
- **Status**: PLANNED (Dispatched in parallel with implementation milestones).

---

## Interface Contracts

### Tokenizer Contract (`helios.tokenizer`)
```python
class HeliosTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert input text string to token ID list."""
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Convert token ID list back to string."""
        ...

    def save_pretrained(self, save_directory: str) -> None:
        """Export tokenizer.json, tokenizer_config.json, special_tokens_map.json."""
        ...

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "HeliosTokenizer":
        """Load tokenizer from directory."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including special tokens."""
        ...

    @property
    def special_tokens(self) -> dict[str, int]:
        """Return mapping of special token strings to token IDs."""
        ...
```

### Model Contract (`helios.model`)
```python
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class HeliosConfig:
    vocab_size: int = 32000
    d_model: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 768
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

class HeliosForCausalLM(nn.Module):
    def __init__(self, config: HeliosConfig):
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | dict | None]:
        """
        Returns dict with:
        - "logits": torch.Tensor of shape (batch, seq_len, vocab_size)
        - "loss": scalar cross-entropy loss if labels provided, else None
        - "kv_cache": updated cache dict if use_cache is True
        """
        ...

def count_parameters(model: nn.Module) -> dict[str, int]:
    """Returns {"total_parameters": int, "trainable_parameters": int}."""
    ...
```

### Hierarchical Memory Architecture Contract (`helios.memory`)
```python
from dataclasses import dataclass
from typing import Any

@dataclass
class MemoryItem:
    id: str
    content: str
    tier: str  # "working" | "episodic" | "semantic" | "procedural"
    metadata: dict[str, Any]
    score: float = 0.0

class HMAController:
    def store_episodic(self, session_id: str, role: str, content: str) -> str:
        """Persist interaction turn to SQLite store."""
        ...

    def index_semantic(self, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        """Add document to dense & sparse semantic retrieval index."""
        ...

    def query(self, query_text: str, top_k: int = 5) -> list[MemoryItem]:
        """Execute hybrid dense + sparse search with Reciprocal Rank Fusion."""
        ...

    def get_working_context(self, session_id: str, max_tokens: int) -> list[dict[str, str]]:
        """Retrieve bounded active working memory turns."""
        ...
```

### Training Controller Contract (`helios.training`)
```python
class TrainingController:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, config: "TrainingConfig"):
        ...

    def save_checkpoint(self, checkpoint_path: str, dataloader_cursor: dict) -> None:
        """Atomically save weights, optimizer, scheduler, RNGs, and data cursor."""
        ...

    def load_checkpoint(self, checkpoint_path: str) -> dict:
        """Restore exact 6-state execution environment and return dataloader cursor."""
        ...

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Execute microbatch step, gradient clipping, and return step loss."""
        ...
```

---

## Code Layout
```
helios/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   └── research/
│       └── model_analysis.md
├── helios/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── embeddings.py
│   │   ├── rotary_embedding.py
│   │   ├── normalization.py
│   │   ├── attention.py
│   │   ├── feed_forward.py
│   │   ├── transformer_block.py
│   │   ├── transformer.py
│   │   └── parameter_inspector.py
│   ├── tokenizer/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── bpe.py
│   │   ├── unigram.py
│   │   ├── special_tokens.py
│   │   └── evaluation.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── working.py
│   │   ├── episodic.py
│   │   ├── semantic.py
│   │   ├── procedural.py
│   │   └── controller.py
│   └── training/
│       ├── __init__.py
│       ├── config.py
│       ├── checkpoint.py
│       ├── controller.py
│       └── smoke_test.py
├── tests/
│   ├── unit/
│   │   ├── test_tokenizer.py
│   │   ├── test_model.py
│   │   ├── test_memory.py
│   │   └── test_training.py
│   ├── integration/
│   │   ├── test_e2e_inference.py
│   │   └── test_training_resumption.py
│   └── e2e/
│       ├── runner.py
│       ├── test_tier1_features.py
│       ├── test_tier2_boundaries.py
│       ├── test_tier3_combinations.py
│       └── test_tier4_scenarios.py
├── pyproject.toml
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
└── CITATION.cff
```
