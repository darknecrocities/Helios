# HELIOS Continuous Training Pipeline — Reproducibility Report

**Author**: Senior ML Infrastructure & Reliability Engineering Team  
**Date**: November 9, 2026  
**Subject**: Data Provenance, Cryptographic Checkpointing & Evaluation Standards  
**Classification**: Research Reproducibility Specification  

---

## 1. Overview & Reproducibility Standard

To ensure that every HELIOS checkpoint is reproducible, verifiable, and free of benchmark contamination, the pipeline establishes an end-to-end audit trail anchored by cryptographic hashes, deterministic PRNG state tracking, and append-only documentation tables.

---

## 2. Cryptographic Checksum Anchoring

Every checkpoint saved to persistent storage includes an immutable `SHA256SUMS` file and `manifest.json`:

```text
checkpoints/step_00000050/
├── state.pt           # Exact 6-state PyTorch dictionary (model, opt, sched, scaler, rng, cursor)
├── manifest.json      # Structured Pydantic execution manifest (step, loss, cluster_id, size)
└── SHA256SUMS         # Streaming SHA-256 digest of state.pt and manifest.json
```

Before loading any checkpoint, `AtomicCheckpointManager.verify_checkpoint()` computes the SHA-256 checksum of `state.pt` in 1 MB stream chunks and verifies it against `SHA256SUMS`. Checkpoints with mismatched digests are flagged as corrupted and rejected.

---

## 3. Deterministic PRNG State Restoration

Unlike toy implementations that only save scalar step numbers, HELIOS records and restores the random generator states for four independent subsystems:
1. `torch.get_rng_state()`: PyTorch CPU tensor generator.
2. `torch.cuda.get_rng_state_all()`: GPU CUDA generators (when running on Colab or Linux CUDA).
3. `np.random.get_state()`: NumPy PRNG state (used in batch sampling and data augmentation).
4. `random.getstate()`: Python standard library random state (used in curriculum partitioning).

Upon pipeline resumption, these four states are restored before loading the next microbatch, ensuring exact sequence reproducibility across restarts.

---

## 4. Dataset Provenance & Curriculum Versioning

Data ingestion enforces cryptographic traceability across all four curriculum stages:

| Stage | Domain / Source | Licensing | Pinned Revision | Ingestion Mode |
| :--- | :--- | :--- | :--- | :--- |
| **Stage A (40%)** | `HuggingFaceFW/fineweb` | ODC-By | `main` (commit pinned) | Bounded stream with PII redaction |
| **Stage B (30%)** | OpenWebMath, arXiv, StarCoder | Apache-2.0 / CC-BY | `v1.2` / Pinned | Regex code and LaTeX parser |
| **Stage C (20%)** | Ultrachat, Capybara, Dolly | MIT / Apache-2.0 | `2024.1` | Multi-turn dialogue formatter |
| **Stage D (10%)** | BFCL MCP Suite, LongBench | Apache-2.0 | `2.1.0` | Function-calling schema evaluator |

Every completed cluster appends an immutable entry to:
- `docs/DATASET_CATALOG.md`: Cluster ID, documents accepted, tokens processed, and content SHA-256.
- `docs/DATASET_PROVENANCE.md`: Licensing and upstream source repository URLs.
- `docs/DATASET_PROCESSING_HISTORY.md`: Rejection breakdown and sanitization statistics.

---

## 5. Append-Only Training & Benchmark Registries

Model progression is captured in real time across four historical records:
1. `docs/TRAINING_HISTORY.md`: Optimizer step, cross-entropy loss, perplexity, and learning rate curves.
2. `docs/CHECKPOINT_HISTORY.md`: Checkpoint ID, model tier, file size (MB), and SHA-256 digest.
3. `docs/BENCHMARK_HISTORY.md`: Evaluated task scores (SciQ, GSM8K, MMLU, HumanEval) relative to open-weight baselines (Qwen2.5 0.5B, Llama 3.2 1B/3B, Gemma 2 2B).
4. `docs/MODEL_EVOLUTION.md`: Parameter scale transitions from Nano 18M to Flagship 5.8B.
