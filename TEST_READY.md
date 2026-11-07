# TEST_READY: Helios Opaque-Box E2E Test Suite (Tiers 1-4)

**Status**: READY FOR IMPLEMENTATION & EXECUTION  
**Version**: 0.1.0  
**Architect & Lead**: Arron Kian Parejas (`parejasarronkian@gmail.com`)  
**Repository**: `https://github.com/darknecrocities/Helios`  
**Test Suite Directory**: `tests/e2e/`  

---

## 1. Executive Summary & Verification Philosophy

This document formally specifies the **Opaque-Box End-to-End (E2E) Test Suite** for the Helios Foundation Model ecosystem as defined in `PROJECT.md` (Milestone 6) and `TEST_INFRA.md`.

The test suite operates under a strict **opaque-box requirement-driven testing contract**:
1. **Zero Coupling to Private Internals**: Tests verify behavior strictly through public APIs (`HeliosForCausalLM`, `HeliosBPE`, `HeliosUnigram`, `HMAController`, `Trainer`, `TrainingConfig`, `ParameterInspector`) and CLI entry points.
2. **Progressive Testability Hierarchy**:
   - **Tier 1 (Feature Equivalence)**: Happy-path verification of all 37 inventoried features across 6 domains.
   - **Tier 2 (Boundary & Extreme Limits)**: Stress-testing zero/max sequence lengths, empty inputs, numerical extremes, memory saturation, and signal boundaries.
   - **Tier 3 (Combinatorial & Pairwise Interactions)**: Verification of multi-subsystem coupling (Tokenizer + Model, Model + KV Cache, Memory + Retrieval, Controller + Checkpoints).
   - **Tier 4 (Real-World Application Scenarios)**: Full lifecycle workflows (Greenfield audit, bilingual chat, pretraining interruption and resumption, deployment exports).
3. **Execution Safety**: Tests execute on CPU or Apple Silicon MPS in $< 15$ seconds without external network calls, GPU cluster dependencies, or persistent filesystem mutation.

---

## 2. Test Inventory & Coverage Matrix

| Test Tier | Source Test File | Target Domain / Features | Test Count (Parametrized) | Minimum Threshold | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Tier 1** | `tests/e2e/test_tier1_features.py` | All 37 Features across 6 Domains (F1–F37) | **37** | 37 | READY |
| **Tier 2** | `tests/e2e/test_tier2_boundaries.py` | Boundary Values, Extreme Limits, Tampering | **40** | 40 | READY |
| **Tier 3** | `tests/e2e/test_tier3_combinations.py` | Pairwise Cross-Feature Interactions | **14** | 14 | READY |
| **Tier 4** | `tests/e2e/test_tier4_scenarios.py` | Real-World Application Scenarios (Scenarios 1–6) | **6** | 6 | READY |
| **TOTAL** | `tests/e2e/` | **Complete E2E Requirement Coverage** | **97** | **97** | **PASS GATE** |

---

## 3. Detailed Feature Mapping (Tier 1)

| Feature # | Feature Name | Test Function | Verified Contract |
| :--- | :--- | :--- | :--- |
| **F1** | Git Remote & Branch Setup | `test_f1_git_remote_and_branch_setup` | Pairs with `github.com/darknecrocities/Helios`, branch `main` |
| **F2** | Python Packaging & Config | `test_f2_python_packaging_and_config` | `pyproject.toml` PEP 517/621 metadata, dependencies, linters |
| **F3** | Open Source Governance Files | `test_f3_open_source_governance_files` | README, LICENSE (Apache/MIT), CONTRIBUTING, CoC, SECURITY, CITATION |
| **F4** | Model Problem Analysis | `test_f4_research_model_problem_analysis` | Astra, Opus, Gemini, Muse analysis and Helios mitigations |
| **F5** | GitHub Actions CI Pipeline | `test_f5_github_actions_ci_pipeline` | Automated test, lint, and type-check jobs in `.github/workflows/ci.yml` |
| **F6** | Byte-Level BPE Tokenizer | `test_f6_byte_level_bpe_engine` | Lossless UTF-8 roundtrip encoding with 256-byte base vocabulary |
| **F7** | Unigram Tokenizer | `test_f7_unigram_tokenizer_engine` | SentencePiece-style Viterbi lattice scoring and decoding |
| **F8** | Special Tokens Contract | `test_f8_special_tokens_contract` | BOS, EOS, PAD, chat turn headers, tool call/return, refusal tokens |
| **F9** | Multilingual & Code Fertility | `test_f9_multilingual_and_code_fertility` | Fertility metrics calculation across English, Filipino, and Python code |
| **F10** | Tokenizer HF Serialization | `test_f10_tokenizer_hf_serialization` | `save_pretrained` / `from_pretrained` roundtrip with JSON configs |
| **F11** | Model Configuration | `test_f11_model_configuration` | `HeliosConfig` parameter validation, presets, JSON export |
| **F12** | Token Embeddings & Weight Tying | `test_f12_token_embeddings_and_weight_tying` | Weight tying identity between `lm_head` and `embed_tokens` |
| **F13** | Rotary Position Embeddings (RoPE) | `test_f13_rotary_position_embeddings` | Relative position representation, complex/half rotation |
| **F14** | Pre-RMSNorm Normalization | `test_f14_pre_rmsnorm_normalization` | Unit variance normalization with numerical stability eps |
| **F15** | GQA & Dynamic KV Cache | `test_f15_gqa_and_dynamic_kv_cache` | 3:1 GQA query:kv head ratio, stateful incremental `KVCache` |
| **F16** | SwiGLU Gated Feed-Forward | `test_f16_swiglu_gated_feed_forward` | Swish-gated linear unit with $d_{ffn} \approx \frac{8}{3} d_{model}$ expansion |
| **F17** | Decoder-Only Transformer LM | `test_f17_decoder_only_transformer_lm` | `HeliosForCausalLM` forward pass, logits, causal cross-entropy loss |
| **F18** | Parameter Inspector Tool | `test_f18_parameter_inspector_tool` | Non-duplicating tied parameter counter and memory footprint estimation |
| **F19** | Nano Target Sizing Verification | `test_f19_nano_target_sizing_verification` | `nano-15m`, `nano-20m`, `nano-25m`, `nano-35m` fit in 10M–40M envelope |
| **F20** | Working Memory Buffer | `test_f20_working_memory_buffer` | In-memory bounded sliding window context manager |
| **F21** | Episodic Memory Store | `test_f21_episodic_memory_store` | SQLite persistent store with WAL mode, sessions, and timestamps |
| **F22** | Semantic Memory (Hybrid) | `test_f22_semantic_memory_hybrid` | Dense cosine/vector search combined with sparse BM25 indexing |
| **F23** | Reciprocal Rank Fusion (RRF) | `test_f23_reciprocal_rank_fusion` | Hybrid ranking merge with $k=60$ and threshold filtering |
| **F24** | Procedural Memory & Skills | `test_f24_procedural_memory_skill_registry` | Tool schema registration, parameter validation, capability lookup |
| **F25** | Memory Weight Isolation Barrier | `test_f25_memory_governance_weight_isolation` | Zero model weight drift during external memory reads and writes |
| **F26** | Training Config & Loss Tracking | `test_f26_training_config_and_loss_tracking` | `TrainingConfig`, gradient accumulation, loss tracking |
| **F27** | L2 Gradient Clipping & Abort | `test_f27_l2_gradient_clipping_and_loss_abort` | Gradient norm clipping, NaN/Inf anomaly detection |
| **F28** | Atomic Checkpoint Management | `test_f28_atomic_checkpoint_management` | Atomic rename write (`.tmp` -> `.pt`) with SHA-256 integrity verification |
| **F29** | POSIX Signal Trapping | `test_f29_posix_signal_trapping` | `SIGINT`/`SIGTERM` trapping on microbatch boundaries |
| **F30** | Exact 6-State Resumption | `test_f30_exact_6_state_resumption_engine` | Restores weights, AdamW optimizer, scheduler, RNGs, cursors, metrics |
| **F31** | Overfit-One-Batch Smoke Test | `test_f31_overfit_one_batch_smoke_test` | Drives synthetic batch loss to near-zero ($< 0.05$) in $\le 150$ steps |
| **F32** | Opaque-Box E2E Suite Readiness | `test_f32_opaque_box_e2e_suite_readiness` | Verifies presence and structure of `tests/e2e/runner.py` |
| **F33** | Adversarial Coverage Readiness | `test_f33_adversarial_coverage_hardening_readiness` | Verifies presence and structure of `tests/adversarial/` (Tier 5) |
| **F34** | Ollama Deployment Modelfile | `test_f34_ollama_deployment_modelfile_generation` | Modelfile generation for `parejasarronkian/Helios` |
| **F35** | Hugging Face Hub Export | `test_f35_huggingface_hub_export_workflow` | Model export bundle for `Darknecrocities/Helios` |
| **F36** | Interactive Live Testing CLI | `test_f36_interactive_live_testing_cli` | Typer CLI runner for `version`, `inspect`, `smoke_test`, `roadmap` |
| **F37** | One Touch / External Disk Mirroring | `test_f37_onedrive_disk_preservation_mirroring` | Checkpoint staging to external disk directory preserving local disk space |

---

## 4. Test Execution & CLI Runner Instructions

### Standalone Runner (Recommended)
```bash
# Execute entire E2E suite (Tiers 1-4)
python -m tests.e2e.runner --tier all

# Execute specific tier with verbose reporting
python -m tests.e2e.runner --tier 1 -v

# Execute with automated JSON and Markdown report export
python -m tests.e2e.runner --tier all --json-report artifacts/e2e_results.json --markdown-report artifacts/e2e_summary.md

# Stop immediately on first failure
python -m tests.e2e.runner -x
```

### Direct Pytest Invocations
```bash
# Run all E2E tests
pytest tests/e2e/ -v

# Run Tier 1 feature tests
pytest tests/e2e/test_tier1_features.py -v

# Run Tier 2 boundary tests
pytest tests/e2e/test_tier2_boundaries.py -v

# Run Tier 3 combination tests
pytest tests/e2e/test_tier3_combinations.py -v

# Run Tier 4 real-world application scenarios
pytest tests/e2e/test_tier4_scenarios.py -v
```

---

## 5. Exit Code Semantics

The test runner adheres to strict process exit code conventions:
- **`0` (Success)**: 100% of selected test cases passed with zero failures or errors.
- **`1` (Test Failure)**: One or more test cases failed assertions or raised unhandled exceptions.
- **`2` (Invocation Error)**: CLI syntax error, invalid `--tier` choice, or configuration failure.

---

## 6. Performance Benchmarks & Quality Gates

- **Execution Time Budget**: Total suite execution must complete within 20.0 seconds on standard 8-core CPU or Apple Silicon M-series.
- **Source Code Physical Ceiling**: Every test file and runner module strictly complies with the physical limit of $< 500$ lines.
- **Memory Footprint**: Transient peak memory $< 512$ MB on CPU.
- **Pass Rate Gate**: 100.0% pass rate mandatory for Milestone 6 completion.
