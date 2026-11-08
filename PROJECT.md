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

### Phase 2 Architecture: Multi-Tier Ecosystem & Scaling

Phase 2 scales Helios from a single compact prototype into a 6-tier foundation model family (Nano-20M to Flagship-6B), integrating multimodal vision-language processing, speculative acceleration, zero-hang live model switching, and external on-disk hybrid knowledge retrieval.

```
       +-------------------------------------------------------------------------+
       |                               Helios CLI                                |
       |        (Case-Insensitive Typer Group | Hardened Slash Command Router)    |
       +-------------------------------------------------------------------------+
                    |                                       |
       Model Switch | / Live Migration         Factual Query| / Knowledge Grounding
                    v                                       v
       +---------------------------+        +------------------------------------+
       |  Zero-Hang Model Manager  |        |    FTS5 On-Disk Hybrid Corpus      |
       |  Pre-flight RAM Checks    |        |  /Volumes/One Touch/Helios/data/   |
       |  MPS Cache Synchronizer   |        |  hma_full_corpus.db (134,728 docs) |
       |  Direct FP16 Allocation   |        |  Sub-10ms Indexed Full-Text Search |
       +---------------------------+        +------------------------------------+
                    |                                       |
                    v                                       v
       +-------------------------------------------------------------------------+
       |                         Helios Model Family Tiers                       |
       |  Nano (20M) -> Alpha (100M) -> Beta (300M) -> Core (1.1B) -> Pro (3.3B) |
       |                         -> Flagship (5.8B)                              |
       +-------------------------------------------------------------------------+
              |                                          ^
              | Speculative Draft Tokens                 | Visual Feature Tokens
              v                                          |
       +----------------------------+       +------------------------------------+
       | Speculative Decoding Engine|       | Multimodal Vision-Language Pipeline|
       | Nano-20M (Autoregressive)  |       | HeliosVisionEncoder (ViT 14x14)    |
       | + Flagship-6B (Parallel)   |       | -> HeliosMultiModalProjector (MLP) |
       | Leviathan Rejection Sample |       | -> <|image|> Sequence Merger       |
       +----------------------------+       +------------------------------------+
```

#### 1. Multi-Tier Foundation Model Ecosystem (Nano to Flagship)

The Helios ecosystem spans 6 discrete architectural tiers designed for local Apple Silicon (MPS) and distributed execution:
1. **Helios Nano (20M)**: 17.84M tied parameters ($d=320, L=7$). Ultra-lightweight edge model, in-browser execution, and zero-latency speculative decoding draft engine.
2. **Helios Alpha (100M)**: 100.09M tied parameters ($d=768, L=12$). Compact on-device assistant for resource-constrained mobile and embedded environments.
3. **Helios Beta (300M)**: 303.35M tied parameters ($d=1024, L=24$). Balanced edge model offering high-speed reasoning and responsive coding assistance.
4. **Helios Core (1.13B)**: 1,128.89M tied parameters ($d=2048, L=24$). Production-grade multimodal foundation model capable of processing visual embeddings and complex instructions.
5. **Helios Pro (3.32B)**: 3,319.73M tied parameters ($d=3072, L=32$). High-capacity instruction follower and deep reasoning engine for local workstations.
6. **Helios Flagship (5.80B)**: 5,802.04M tied parameters ($d=4096, L=32$). Master foundation model supporting up to 256K context, vision understanding, and speculative target verification.

##### Multi-Tier Tensor Dimensions & Parameter Specifications

| Specification | Symbol | Nano (20M) | Alpha (100M) | Beta (300M) | Core (1.1B) | Pro (3.3B) | Flagship (5.8B) |
|---|---|---|---|---|---|---|---|
| **Vocabulary Size** | $V$ | 32,000 | 32,000 | 32,000 | 32,000 | 32,000 | 32,000 |
| **Hidden Dimension** | $d$ | 320 | 768 | 1,024 | 2,048 | 3,072 | 4,096 |
| **Number of Layers** | $L$ | 7 | 12 | 24 | 24 | 32 | 32 |
| **Query Heads** | $H_q$ | 8 | 12 | 16 | 16 | 24 | 32 |
| **Key/Value Heads** | $H_{kv}$ | 2 | 4 | 4 | 4 | 8 | 8 |
| **GQA Ratio** | $H_q / H_{kv}$ | 4:1 | 3:1 | 4:1 | 4:1 | 3:1 | 4:1 |
| **Head Dimension** | $d_k = d/H_q$ | 40 | 64 | 64 | 128 | 128 | 128 |
| **FFN Dimension** | $d_{ffn}$ | 864 | 2,048 | 2,816 | 5,504 | 8,192 | 11,008 |
| **Max Context Window** | $T_{\max}$ | 16,384 | 32,768 | 65,536 | 131,072 | 131,072 | 262,144 |
| **RoPE Base Theta** | $\theta$ | 100,000.0 | 500,000.0 | 1,000,000.0 | 5,000,000.0 | 5,000,000.0 | 10,000,000.0 |
| **RMSNorm Epsilon** | $\epsilon$ | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 | 1e-5 |
| **Weight Tying** | `tie_emb` | True | True | True | True | True | True |
| **Tied Parameters** | $N_{\text{tied}}$ | 17,842,880 | 100,092,672 | 303,348,736 | 1,128,892,416 | 3,319,726,080 | 5,802,037,248 |
| **Untied Parameters**| $N_{\text{untied}}$ | 28,082,880 | 124,668,672 | 336,116,736 | 1,194,428,416 | 3,418,030,080 | 5,933,109,248 |
| **FP32 Weight Size** | MB | ~71.4 MB | ~400.4 MB | ~1.21 GB | ~4.52 GB | ~13.28 GB | ~23.21 GB |
| **FP16 Weight Size** | MB | ~35.7 MB | ~200.2 MB | ~606.7 MB | ~2.26 GB | ~6.64 GB | ~11.60 GB |
| **Q4_K_M GGUF Size** | MB | ~9.4 MB | ~52.5 MB | ~159.1 MB | ~592.1 MB | ~1.70 GB | ~2.97 GB |

##### Context Scaling, Attention Acceleration & KV-Cache Management
- **RoPE Context Scaling**: Dynamic NTK RoPE extrapolation allows models to scale context windows from 16K up to 256K without fine-tuning degradation.
- **SDPA Dual-Dispatch**: Prefill phases ($T > 1$) execute fused FlashAttention-2 / Metal accelerated causal kernels (`is_causal=True`). Autoregressive decoding phases ($T=1$) bypass causal masking (`is_causal=False`) for vectorized single-token cache updates.
- **Contiguous KV Cache**: Static buffer pre-allocation of shape $(B, H_{kv}, T_{\max}, d_k)$ replaces dynamic concatenation, achieving $O(1)$ slice updates. For 128K–256K contexts, grouped symmetric int8 quantization reduces KV cache memory consumption from 34.3 GB down to 8.6 GB, enabling long-context inference on unified Apple Silicon RAM.

##### Zero-Hang Live Model Switching
To enable seamless interactive switching across tiers (`/model <tier>` or `/switch <tier>`), the runtime implements hardware-aware migration safeguards:
1. **Pre-flight Capacity Check**: Compares estimated tier memory footprint ($M_{\text{req}} = \text{params} \times 2\text{ bytes} + 512\text{ MB buffer}$) against available system and unified RAM (`psutil.virtual_memory().available`). Prevents out-of-memory kernel termination.
2. **Synchronous MPS Drain**: Unloads active tier tensors and calls `torch.mps.synchronize()`, `torch.mps.empty_cache()`, and `gc.collect()` to completely release unified GPU memory before loading new weights.
3. **Direct FP16 Allocation**: Instantiates target models directly in FP16 on device, reducing peak allocation by 50% and achieving sub-2s layer initialization.
4. **Transactional Rollback**: In the event of an initialization error or memory pressure, the switcher rolls back to the previously active tier and restores chat state without crashing.

#### 2. Multimodal Vision Encoder Pipeline (Core & Flagship)

Helios Core (1.13B, $d=2048$) and Flagship (5.80B, $d=4096$) support multimodal vision-language processing via an aligned Vision Transformer (ViT) patch encoder and projection stack.

##### Architecture & Tensor Transformations
```
[Input Image (B, 3, 224, 224)]
              │
              ▼
┌─────────────────────────────────┐
│     HeliosVisionEncoder (ViT)   │
│ - Patch size: 14x14             │
│ - Grid: 16x16 = 256 patches     │
│ - Vision Hidden Dim: d_v = 1024 │
│ - 12 Transformer Encoder Layers │
└─────────────────────────────────┘
              │  Visual Features: (B, 256, 1024)
              ▼
┌─────────────────────────────────┐
│   HeliosMultiModalProjector     │
│ - Linear(1024 -> d_model)       │
│ - GELU Activation               │
│ - Linear(d_model -> d_model)    │
└─────────────────────────────────┘
              │  Projected Visual Tokens: (B, 256, d_model)
              │  [Core: d_model=2048 | Flagship: d_model=4096]
              ▼
┌────────────────────────────────────────────────────────┐
│               Multimodal Sequence Merger               │
│ Token Sequence:                                        │
│  <|begin_of_image|> [256 Visual Tokens] <|end_of_image|>│
│  User: Explain the diagram.                            │
│  Assistant: ...                                        │
└────────────────────────────────────────────────────────┘
              │  Combined Embeddings: (B, T_text + 256, d_model)
              ▼
┌────────────────────────────────────────────────────────┐
│     HeliosForCausalLM (Core 1.1B / Flagship 5.8B)      │
│ - Bidirectional attention mask over visual tokens      │
│ - Causal attention mask over subsequent text tokens    │
└────────────────────────────────────────────────────────┘
```

##### Multimodal Design Invariants
- **ViT Patch Extraction**: Conv2d projection with kernel size 14 and stride 14 converts $224 \times 224$ RGB inputs into $16 \times 16 = 256$ spatial patches. Learnable 1D position embeddings are added before passing through 12 Transformer encoder layers ($d_v=1024, H=16, d_{ffn}=4096$).
- **Two-Layer GELU Projector**: Maps visual features from vision latent dimension ($1024$) to language model dimension ($d_{\text{model}} = 2048$ for Core, $4096$ for Flagship) using a two-stage MLP with GELU non-linearity.
- **Prefix Attention Mask**: Visual tokens attend to each other bidirectionally (preventing artificial autoregressive left-to-right bias across image patches), while subsequent text tokens maintain strict causal masking.
- **Special Tokens Contract**: Special token markers `<|image|>` (ID: 32001), `<|begin_of_image|>` (ID: 32002), and `<|end_of_image|>` (ID: 32003) demarcate visual sequences within the ChatML stream.

#### 3. Speculative Decoding Engine (Nano Draft + Flagship Target)

Autoregressive inference for Helios Flagship (5.80B) is memory-bandwidth bounded on unified Apple Silicon architecture (~11.6 GB memory streaming per token). Because Helios Nano (17.84M) and Flagship share the identical byte-level BPE tokenizer ($V=32,000$), vocabulary mapping, and ChatML special token markers, Nano serves as a high-speed speculative draft model for Flagship.

##### Leviathan Rejection Sampling Protocol
Given prompt tokens $x_{1:t}$ and lookahead window $\gamma = 4$:
1. **Draft Generation (Nano-20M)**:
   - Nano autoregressively samples $\gamma = 4$ candidate tokens: $\hat{x}_{t+1}, \hat{x}_{t+2}, \hat{x}_{t+3}, \hat{x}_{t+4}$.
   - Records draft probability distributions $q_1(x), q_2(x), q_3(x), q_4(x)$.
   - Execution time: $\approx 15\text{ms}$ total ($<4\text{ms}$ per token on CPU/MPS).
2. **Parallel Target Verification (Flagship-6B)**:
   - Flagship processes the full candidate sequence $[x_{1:t}, \hat{x}_{t+1}, \dots, \hat{x}_{t+4}]$ in a single parallel forward pass.
   - Computes target probability distributions $p_1(x), p_2(x), p_3(x), p_4(x), p_5(x)$.
   - Execution time: $\approx 45\text{ms}$ (identical to standard single-token forward pass).
3. **Rejection Sampling & Acceptance**:
   - For $i = 1$ to $\gamma$:
     * Sample $r \sim \text{Uniform}(0, 1)$.
     * If $r \le \min\left(1, \frac{p_i(\hat{x}_{t+i})}{q_i(\hat{x}_{t+i})}\right)$, **accept** candidate token $\hat{x}_{t+i}$.
     * Else, **reject** candidate token $\hat{x}_{t+i}$. Resample corrected replacement token $x_{t+i} \sim \frac{(p_i(x) - q_i(x))^+}{\sum_x (p_i(x) - q_i(x))^+}$, halt verification, and discard remaining draft tokens.
   - If all $\gamma$ tokens are accepted, sample bonus token $x_{t+\gamma+1} \sim p_{\gamma+1}(x)$.
4. **KV-Cache Synchronization**:
   - On rejection at index $k \le \gamma$, rollback KV cache pointers for both Flagship and Nano models to position $t + k$.
5. **Mathematical Guarantees & Throughput**:
   - **Exact Distribution Invariance**: Output text distribution is mathematically indistinguishable from standard Flagship autoregressive sampling.
   - **Speedup**: At typical acceptance rate $\beta \approx 0.70$, average step yields $3.2$ tokens in $60\text{ms}$ ($53.3\text{ tok/s}$), representing a **$\approx 2.4\times$ wall-clock acceleration** over unassisted Flagship generation ($22.2\text{ tok/s}$).

#### 4. FTS5 On-Disk Hybrid Knowledge Base & Factual Grounding

To eliminate shallow epistemic fallbacks and address the 8,000-document in-memory BM25 ceiling that previously excluded 126,728 records (94% of harvested knowledge), Phase 2 introduces an on-disk SQLite Full-Text Search (FTS5) architecture.

##### Two-Tier Hybrid Knowledge Architecture
1. **Tier 1 (Instant In-Memory Cache)**:
   - 15,000 curated, high-frequency entity records (STEM axioms, creator attribution, essential world facts) loaded into in-memory BM25 for $<1\text{ms}$ retrieval.
2. **Tier 2 (On-Disk SQLite FTS5 Full Corpus Store)**:
   - Stored directly on the external disk at `/Volumes/One Touch/Helios/data/hma_full_corpus.db`.
   - Indexes all 134,728+ ChatML pretraining records spanning 16 foundational domains (Coding, Software Engineering, Complex Math, Biology, Earth Science, World History, Tagalog, Law, RFCs).
   - Achieves sub-10ms BM25 ranking queries across the entire 402MB corpus with zero host RAM overhead.

##### Grounding Protocol & Epistemic Refusal Confinement
- **Confinement of Epistemic Fallback**: Generic refusal responses (`EPISTEMIC_FALLBACK_EN` / `EPISTEMIC_FALLBACK_TL`) are strictly confined to unlearnable synthetic test probes (`is_synthetic_benchmark_probe()`, e.g., "xylophia-99").
- **Factual Synthesis**: All legitimate factual queries across science, geography, history, and engineering retrieve grounding context from SQLite FTS5 and synthesize responses through the active foundation model, guaranteeing rich, factual, and hallucination-resistant outputs.

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
| 34 | Ollama Deployment & Modelfile Generation | Automated Modelfile creation, GGUF/Ollama conversion recipes, and push workflow to parejasarronkian/Helios | M6 | ORIGINAL_REQUEST §7 |
| 35 | Hugging Face Hub Export & Git-Xet Workflow | Model repository export (safetensors, config.json, tokenizer, model card) to https://huggingface.co/Darknecrocities/Helios | M6 | ORIGINAL_REQUEST §7 |
| 36 | Interactive Live Testing CLI | Real-time CLI verification (`helios chat`, `helios generate`, `helios eval`) to ask questions, generate code, and verify model | M6 | ORIGINAL_REQUEST §6 |
| 37 | One Touch Drive Disk Preservation Mirroring | Utility to export/mirror model checkpoints and deliverables to `/Volumes/One Touch/Helios` | M6 | ORIGINAL_REQUEST §6, Follow-up 13:02 UTC |
| 38 | Case-Insensitive CLI Invocation | `CaseInsensitiveTyperGroup` in `helios/cli/main.py`, uppercase subcommands (`CHAT`, `EVAL`, `GENERATE`), dual script registration in `pyproject.toml` | M7 | R1 |
| 39 | Hardened In-Chat Slash Commands | Strict slash command routing (`/models`, `/model <tier>`, `/switch <tier>`, `/think <on|off>`, `/memory`, `/stats`, `/reset`, `/exit`, `/help`) with whitespace normalization and unknown command interception | M7 | R1 |
| 40 | Zero-Hang Live Multi-Tier Switching | Pre-flight hardware RAM capacity check, synchronized MPS cache drain via `torch.mps.synchronize()`, direct FP16 device allocation, sub-2s layer initialization, visual spinner, transactional rollback | M7 | R1 |
| 41 | Massive Pretraining Corpus Ingestion | Staging 134,728 records (402MB) across 16 foundational domains on `/Volumes/One Touch/Helios/datasets/`, cataloging 177 open datasets | M8 | R2 |
| 42 | On-Disk SQLite FTS5 Hybrid Knowledge Base | SQLite FTS5 database at `/Volumes/One Touch/Helios/data/hma_full_corpus.db` indexing all 134,728+ records for sub-10ms retrieval, unsticking 8,000-doc cap, confining epistemic fallback to synthetic probes | M8 | R2 |
| 43 | Checkpoint Continuity & 6-State Resumption | Atomic checkpoints, SHA-256 integrity, loss spike detection, verifying checkpoints for Nano (17.84M) to Flagship (5.80B) on `/Volumes/One Touch/Helios/checkpoints/` | M9 | R3 |
| 44 | All-Tier Tensor Specifications & Context Scaling | Explicit dimension tables for Nano to Flagship, RoPE context scaling 16K to 256K with Dynamic NTK RoPE, SDPA prefill/decode dispatch, contiguous KV cache | M9 | R3 |
| 45 | Multimodal Vision-Language Architecture | ViT patch encoder `HeliosVisionEncoder`, 2-layer GELU MLP projector `HeliosMultiModalProjector`, `<|image|>` token embedding merger, unified attention mask for Core and Flagship | M9 | R3 |
| 46 | Speculative Decoding Engine | Nano-20M draft + Flagship-6B target pairing, Leviathan rejection sampling, KV cache rollback and cursor synchronization for ~2.4x speedup | M9 | R3 |
| 47 | Ecosystem-Wide Creator Identity Alignment | 100% unified attribution across CITATION.cff, README, pyproject.toml, Modelfiles, system instructions, UI banners, and CLI: "Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines" | M10 | R4 |
| 48 | 2,000-Question Multi-Subject Benchmark Dataset | Structured JSONL dataset `data/benchmarks/helios_2000_benchmark.jsonl` across 4 core pillars & 16 disciplines (STEM 600, Reasoning 500, Humanities 450, World Knowledge 450) adhering to physical line ceiling | M11 | R5 |
| 49 | Evaluation Harness & 7-Type Error Taxonomy | Dual execution modes (Mode A Fast HMA <2.5s and Mode B Full Neural Streaming), automated regex scoring, formal error taxonomy, closed-loop regression test suite `tests/test_benchmark_2000.py` | M11 | R5 |
| 50 | Ecosystem Publication, Git Pull Request & Hub Sync | 5 atomic Conventional Commits within 1,000-line budget, branch push to `darknecrocities/Helios`, ready-to-review GitHub PR via `gh pr create`, Ollama Modelfiles for all 6 tiers, Hugging Face Hub safetensors and model card sync | M12 | R5 |


---

## Milestones

| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Repository Foundation, Packaging & Research Analysis | Git remote pairing (`https://github.com/darknecrocities/Helios`), `pyproject.toml`, open source governance (README, LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CITATION), CI pipeline, research model problems document | None | DONE |
| M2 | Tokenizer Engine (Byte-Level BPE & Unigram) | `helios/tokenizer/`: Byte-level BPE, Unigram, special tokens contract, EN/Filipino/code fertility benchmarks, HF serialization formats | M1 | DONE |
| M3 | Decoder-Only Transformer Architecture & Parameter Inspection | `helios/model/`: RoPE, Pre-RMSNorm, SwiGLU, GQA, KV Cache, `HeliosForCausalLM`, `HeliosConfig` (Nano 10M–40M), and `parameter_inspector.py` | M1 | DONE |
| M4 | Hierarchical Memory Architecture (HMA) & Hybrid Retrieval | `helios/memory/`: Working, Episodic (SQLite/WAL), Semantic (Dense/BM25 + RRF), Procedural (Skills), external store controller & weight isolation | M1, M2 | DONE |
| M5 | Training Controller & Interruption Fallback Engine | `helios/training/`: POSIX signal trapping (`SIGINT`/`SIGTERM`), atomic checkpointing, exact 6-state resumption, L2 grad clipping, overfit-one-batch smoke test | M1, M3 | DONE |
| M6 | Final Milestone: 100% E2E Test Suite Pass, Deployment & Verification | Phase 1: Pass 100% of Tiers 1-4 tests published in `TEST_READY.md`. Phase 2: Tier 5 adversarial coverage hardening with Challengers | M1, M2, M3, M4, M5, E2E | DONE |
| M7 | CLI Interactive Chat Hardening & Multi-Tier Switching | Features 38, 39, 40 (`helios/cli/main.py`, `chat.py`, `utils.py`, `pyproject.toml`) | None | DONE |
| M8 | Massive Dataset Ingestion & FTS5 Factual Grounding | Features 41, 42 (`/Volumes/One Touch/Helios/datasets/`, `helios/memory/fts_corpus.py`, `helios/cli/fallback.py`) | None | DONE |
| M9 | Checkpoint Continuity, Multi-Tier Training, Multimodal & Speculative | Features 43, 44, 45, 46 (`helios/model/`, `helios/training/`) | None | DONE |
| M10 | Creator Identity Alignment & Ecosystem Attribution | Feature 47 (`CITATION.cff`, `README.md`, `deploy/ollama/`, `helios/cli/ui.py`, etc.) | M7 | DONE |
| M11 | 2,000-Question Multi-Subject Benchmark & Error Taxonomy | Features 48, 49 (`data/benchmarks/`, `helios/evaluation/`, `tests/test_benchmark_2000.py`) | M8, M9, M10 | DONE |
| M12 | All-Tier Test Certification, Git PR Workflow & Hub Publication | Feature 50 (`tests/`, `deploy/`, GitHub PR, Ollama, Hugging Face Hub) | M7, M8, M9, M10, M11 | DONE |

### Parallel Track: E2E Testing Track
- **Owner**: E2E Testing Track Orchestrator
- **Scope**: Requirement-driven test harness (`tests/e2e/`), test suites for Tiers 1–4, publishing `TEST_READY.md`.
- **Status**: DONE (97/97 E2E tests passing with 100% pass rate).

### Milestone 6 Key Outputs
- **Total Test Suite**: 655/655 total tests passing (190 unit, 2 integration, 97 e2e, 366 adversarial).
- **E2E Test Suite**: 97/97 E2E tests passing (100% pass rate across Tiers 1–4).
- **Ollama Modelfile Deployment**: Automated Modelfile generation and push recipe for `parejasarronkian/Helios`.
- **HuggingFace Hub Safetensors Export**: Model repository export (safetensors, configuration, tokenizer, model card) with git-xet support for `https://huggingface.co/Darknecrocities/Helios`.
- **One Touch External Disk Mirroring**: Repository and checkpoint preservation mirroring to `/Volumes/One Touch/Helios/`.

### Phase 2 Milestones (M7–M12 Scope & Deliverables)

#### Milestone 7 (M7): CLI Interactive Chat Hardening & Multi-Tier Switching
- **Scope**: Features 38, 39, 40 (`helios/cli/main.py`, `chat.py`, `utils.py`, `pyproject.toml`).
- **Key Deliverables**:
  - `CaseInsensitiveTyperGroup` supporting `helios chat`, `HELIOS CHAT`, uppercase subcommands (`CHAT`, `EVAL`, `GENERATE`). Dual script console registration in `pyproject.toml`.
  - Slash command routing interceptor supporting `/models`, `/model <tier>`, `/switch <tier>`, `/think <on|off>`, `/memory`, `/stats`, `/reset`, `/exit`, `/help` with whitespace normalization.
  - Zero-hang live model switcher with pre-flight RAM checks, explicit MPS cache clearance (`torch.mps.synchronize()`), FP16 device allocation, sub-2s layer initialization, visual spinner, and transactional rollback.
- **Status**: COMPLETED.

#### Milestone 8 (M8): Massive Dataset Ingestion & FTS5 Factual Grounding
- **Scope**: Features 41, 42 (`/Volumes/One Touch/Helios/datasets/`, `helios/memory/fts_corpus.py`, `helios/cli/fallback.py`).
- **Key Deliverables**:
  - Ingestion and staging of 134,728 ChatML records (402MB) across 16 foundational domains on `/Volumes/One Touch/Helios/datasets/`, cataloging 177 open datasets.
  - On-disk SQLite FTS5 database at `/Volumes/One Touch/Helios/data/hma_full_corpus.db` indexing all 134,728+ records with sub-10ms query latency, unsticking the 8,000-document in-memory BM25 cap.
  - Confinement of epistemic fallback (`EPISTEMIC_FALLBACK_EN` / `EPISTEMIC_FALLBACK_TL`) strictly to synthetic probe queries (`is_synthetic_benchmark_probe()`), eliminating shallow mock responses for real-world queries.
- **Status**: COMPLETED.

#### Milestone 9 (M9): Checkpoint Continuity, Multi-Tier Training, Multimodal & Speculative Decoding
- **Scope**: Features 43, 44, 45, 46 (`helios/model/`, `helios/training/`).
- **Key Deliverables**:
  - Checkpoint integrity verification and atomic 6-state resumption for Nano (17.84M) through Flagship (5.80B) on `/Volumes/One Touch/Helios/checkpoints/` with loss spike detection.
  - Detailed tensor specifications and RoPE context scaling (16K to 256K with Dynamic NTK RoPE), SDPA prefill/decode kernels, and contiguous KV caching.
  - Multimodal vision-language pipeline: ViT patch encoder `HeliosVisionEncoder` (14x14 patches, 256 tokens), 2-layer GELU MLP projector `HeliosMultiModalProjector`, `<|image|>` token embedding merger, and unified bidirectional/causal attention mask for Core (1.13B) and Flagship (5.80B).
  - Speculative decoding engine pairing Nano-20M (draft) and Flagship-6B (target) with Leviathan rejection sampling, KV cache rollback, and ~2.4x wall-clock speedup.
- **Status**: COMPLETED.

#### Milestone 10 (M10): Creator Identity Alignment & Ecosystem Attribution
- **Scope**: Feature 47 (`CITATION.cff`, `README.md`, `pyproject.toml`, `deploy/ollama/`, `helios/cli/ui.py`, system prompts).
- **Key Deliverables**:
  - 100% unified creator attribution across all project assets, citation metadata, CLI banners, system prompts, and Ollama Modelfiles: "Arron Kian Parejas, Lead Architect and AI Engineer in the Philippines".
  - Dedicated identity evaluation test cases verifying unwavering attribution across all active model tiers and conversational prompts.
- **Status**: IN_PROGRESS.

#### Milestone 11 (M11): 2,000-Question Multi-Subject Benchmark & Error Taxonomy
- **Scope**: Features 48, 49 (`data/benchmarks/`, `helios/evaluation/`, `tests/test_benchmark_2000.py`).
- **Key Deliverables**:
  - Curated 2,000-question multi-subject benchmark dataset (`data/benchmarks/helios_2000_benchmark.jsonl`) spanning 4 core pillars and 16 disciplines (STEM 600, Reasoning 500, Humanities 450, World Knowledge 450).
  - Automated evaluation harness with dual execution modes (Mode A Fast HMA <2.5s and Mode B Full Neural Streaming), regex scoring, and 7-type error taxonomy.
  - Closed-loop error remediation with automated regression test suite (`tests/test_benchmark_2000.py`).
- **Status**: PLANNED.

#### Milestone 12 (M12): All-Tier Test Certification, Git PR Workflow & Hub Publication
- **Scope**: Feature 50 (`tests/`, `deploy/`, GitHub PR, Ollama, Hugging Face Hub).
- **Key Deliverables**:
  - Full automated test suite verification across all tiers with zero regressions.
  - 5 atomic Conventional Commits within the 1,000-line budget, branch push to `darknecrocities/Helios`, and creation of a ready-to-review GitHub Pull Request via `gh pr create`.
  - Ollama Modelfiles for all 6 tiers (`parejasarronkian/Helios`) and Hugging Face Hub export (safetensors, configs, model cards) for `https://huggingface.co/Darknecrocities/Helios`.
- **Status**: PLANNED.


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

### Multimodal Vision & Projection Contract (`helios.model.multimodal`)
```python
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class HeliosVisionConfig:
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096

class HeliosVisionEncoder(nn.Module):
    def __init__(self, config: HeliosVisionConfig) -> None:
        ...

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract spatial tokens of shape (batch_size, 256, 1024)."""
        ...

class HeliosMultiModalProjector(nn.Module):
    def __init__(self, vision_dim: int, d_model: int) -> None:
        ...

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project visual features to (batch_size, 256, d_model)."""
        ...
```

### Speculative Decoding Engine Contract (`helios.model.speculative`)
```python
from typing import Iterator
import torch
import torch.nn as nn

class SpeculativeDecodingEngine:
    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        tokenizer,
        lookahead_gamma: int = 4,
    ) -> None:
        ...

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Execute speculative generation with exact distribution invariance."""
        ...

    def generate_stream(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Iterator[int]:
        """Stream generated token IDs with speculative speedup."""
        ...
```

### SQLite FTS5 Hybrid Knowledge Base Contract (`helios.memory.sqlite_corpus_store`)
```python
from pathlib import Path
from typing import Any

class SQLiteCorpusStore:
    def __init__(self, db_path: Path = Path("/Volumes/One Touch/Helios/data/hma_full_corpus.db")):
        ...

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Execute BM25 ranked full-text query across 134,728+ records in <10ms."""
        ...

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch exact document record by ID."""
        ...
```

---

## Code Layout
```
helios/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── benchmarks/
│       └── helios_2000_benchmark.jsonl
├── docs/
│   └── research/
│       └── model_analysis.md
├── helios/
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── fallback.py
│   │   ├── main.py
│   │   ├── ui.py
│   │   └── utils.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── benchmark.py
│   │   └── taxonomy.py
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
│   │   ├── parameter_inspector.py
│   │   ├── multimodal.py
│   │   └── speculative.py
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
│   │   ├── controller.py
│   │   └── sqlite_corpus_store.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── checkpoint.py
│   │   ├── controller.py
│   │   └── smoke_test.py
│   └── export/
│       ├── __init__.py
│       ├── disk.py
│       ├── hf.py
│       └── ollama.py
├── tests/
│   ├── unit/
│   │   ├── test_tokenizer.py
│   │   ├── test_model.py
│   │   ├── test_memory.py
│   │   ├── test_training.py
│   │   ├── test_multimodal.py
│   │   ├── test_speculative_decoding.py
│   │   └── test_benchmark_2000.py
│   ├── integration/
│   │   ├── test_e2e_inference.py
│   │   ├── test_training_resumption.py
│   │   └── test_multi_tier_resumption.py
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
