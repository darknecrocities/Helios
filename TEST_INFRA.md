# E2E Test Infra: Helios Open-Weight LLM

## Test Philosophy
- Opaque-box, requirement-driven. Derived strictly from `ORIGINAL_REQUEST.md` and user-facing specifications without coupling to internal implementation classes or private helpers.
- Methodology: Category-Partition + Boundary Value Analysis (BVA) + Pairwise Combinatorial Testing + Real-World Workload Testing.
- Progressive Testability: Test verifications use standard entry points (CLI invocations, public API functions, serialized files) and simplest available verification channels.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | Git Remote & PR Workflow | ORIGINAL_REQUEST §1 | 5 | 5 | ✓ |
| 2 | Python Packaging & Config | Master Plan §16 | 5 | 5 | ✓ |
| 3 | Open Source Governance Files | ORIGINAL_REQUEST §7 | 5 | 5 | ✓ |
| 4 | Model Problem Analysis | ORIGINAL_REQUEST §2 | 5 | 5 | ✓ |
| 5 | GitHub Actions CI | Master Plan §16 | 5 | 5 | ✓ |
| 6 | Byte-Level BPE Tokenizer | ORIGINAL_REQUEST §4 | 5 | 5 | ✓ |
| 7 | Unigram Tokenizer | Master Plan §5.1 | 5 | 5 | ✓ |
| 8 | Special Tokens Contract | Master Plan §5.2 | 5 | 5 | ✓ |
| 9 | Multilingual & Code Fertility | Master Plan §5.1 | 5 | 5 | ✓ |
| 10 | Tokenizer HF Serialization | Master Plan §14.2 | 5 | 5 | ✓ |
| 11 | Helios Model Configuration | Master Plan §4.1 | 5 | 5 | ✓ |
| 12 | Token Embeddings & Weight Tying | Master Plan §4.1 | 5 | 5 | ✓ |
| 13 | RoPE Position Embeddings | Master Plan §4.1 | 5 | 5 | ✓ |
| 14 | Pre-RMSNorm Normalization | Master Plan §4.1 | 5 | 5 | ✓ |
| 15 | GQA & KV Cache | Master Plan §4.1 | 5 | 5 | ✓ |
| 16 | SwiGLU Gated Feed-Forward | Master Plan §4.1 | 5 | 5 | ✓ |
| 17 | Decoder-Only Transformer LM | Master Plan §4.1 | 5 | 5 | ✓ |
| 18 | Parameter Inspector Tool | ORIGINAL_REQUEST §3 | 5 | 5 | ✓ |
| 19 | Nano Target Sizing (10M–40M) | Master Plan §4.2 | 5 | 5 | ✓ |
| 20 | Working Memory Buffer | ORIGINAL_REQUEST §5 | 5 | 5 | ✓ |
| 21 | Episodic Memory Store | Master Plan §10.1 | 5 | 5 | ✓ |
| 22 | Semantic Memory (Dense + Sparse) | Master Plan §10.1 | 5 | 5 | ✓ |
| 23 | Reciprocal Rank Fusion (RRF) | Master Plan §10.2 | 5 | 5 | ✓ |
| 24 | Procedural Memory & Skill Registry | Master Plan §10.1 | 5 | 5 | ✓ |
| 25 | Memory Governance & Isolation | ORIGINAL_REQUEST §5 | 5 | 5 | ✓ |
| 26 | Training Config & Loss Tracking | Master Plan §7.1 | 5 | 5 | ✓ |
| 27 | L2 Grad Clip & Loss Abort | Master Plan §7.2 | 5 | 5 | ✓ |
| 28 | Atomic Checkpoint Management | Master Plan §7.4 | 5 | 5 | ✓ |
| 29 | POSIX Signal Trapping | ORIGINAL_REQUEST §6 | 5 | 5 | ✓ |
| 30 | Exact 6-State Resumption Engine | ORIGINAL_REQUEST §6 | 5 | 5 | ✓ |
| 31 | Overfit-One-Batch Smoke Test | Master Plan §17 | 5 | 5 | ✓ |

## Test Architecture
- Test runner: `tests/e2e/runner.py` / `pytest tests/e2e/`
- Directory layout:
  - `tests/e2e/test_tier1_features.py`: Happy-path equivalence classes for each feature
  - `tests/e2e/test_tier2_boundaries.py`: Boundary values, extreme sequences, zero/max tokens, signal simulations
  - `tests/e2e/test_tier3_combinations.py`: Cross-feature interactions (Tokenizer + Model, Model + KV Cache, Memory + Retrieval, Controller + Checkpoints)
  - `tests/e2e/test_tier4_scenarios.py`: End-to-end user workflows (Full pretraining smoke test, interrupted & resumed training matching baseline loss, bilingual chat session with HMA memory retrieval)

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full Greenfield Project Verification | F1, F2, F3, F4, F5 | Low |
| 2 | Multilingual & Code Prompt Encoding with Special Tokens | F6, F7, F8, F9, F10 | Medium |
| 3 | Nano-15M / Nano-25M Causal Inference with KV Cache | F11, F12, F13, F14, F15, F16, F17, F18, F19 | High |
| 4 | Multi-Turn Agent Conversation with HMA Retrieval Grounding | F20, F21, F22, F23, F24, F25 | High |
| 5 | Interrupted Training Run with Exact State Resumption & Bitwise Continuity | F26, F27, F28, F29, F30, F31 | High |

## Coverage Thresholds
- N = 31 primary features
- Tier 1: ≥ 5 × 31 = 155 test cases
- Tier 2: ≥ 5 × 31 = 155 test cases
- Tier 3: ≥ 31 pairwise interaction test cases
- Tier 4: ≥ 5 real-world end-to-end application scenarios
- **Total Minimum Threshold**: ~346 test cases
