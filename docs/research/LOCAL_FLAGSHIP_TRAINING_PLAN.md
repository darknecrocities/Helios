# Local Training Plan for Core / Pro / Flagship (and Nano/Alpha/Beta)

**Hardware:** Apple M3, 8 CPU cores, 16 GB RAM, MPS backend (no CUDA).  
**Goal:** Train all six tiers, including Core (1.1B), Pro (3.3B), Flagship (5.8B), without corrupting prior work.

## 1. What fits on this machine

| Tier | Unique params | Full state-dict | Est. full-train memory (fp16 weights+grads+AdamW, no activations) | Fits 16 GB? |
|---|---:|---:|---:|---|
| Nano | 17.8M | 28.1M | ~0.3 GB | yes |
| Alpha | 100.1M | 124.7M | ~1.8 GB | yes |
| Beta | 303.4M | 336.1M | ~5.1 GB | yes (tight) |
| Core | 1.13B | 1.25B | ~12.9 GB | marginal, activations overflow |
| Pro | 3.32B | 3.43B | ~37.5 GB | no |
| Flagship | 5.80B | 5.92B | ~65.3 GB | no |

Conclusion: **Core can only be trained with heavy memory optimization; Pro and Flagship cannot be full-pretrained locally.**

## 2. Strategy by tier

### A. Nano / Alpha / Beta — full pretraining locally
- Continue full-parameter pretraining on the 83.5M-token (target 500M) FineWeb corpus.
- Use MPS, bf16 autocast, gradient checkpointing, gradient accumulation, and the existing resumable lineage runner.
- Estimated throughput after optimization: ~150–250K tokens/min for Nano, ~25–40K for Alpha, ~6–10K for Beta.
- 500M tokens: Nano ~35–55 h, Alpha ~200–330 h, Beta ~830–1400 h. **Only Nano is realistic locally at 500M; Alpha/Beta should use 20–50M tokens locally and scale on cloud.**

### B. Core (1.1B) — memory-optimized local training
- Use gradient checkpointing, 8-bit AdamW (bitsandbytes CPU/offload), micro-batch 1, gradient accumulation 32–64.
- Estimated peak memory ~14–16 GB; feasible only with no other heavy apps.
- Throughput ~2–4K tokens/min on MPS; 500M tokens = 2,000–4,000 h. **Not practical for 500M; run 10–30M tokens locally for validation only.**

### C. Pro (3.3B) and Flagship (5.8B) — cannot full-pretrain locally
Options:
1. **Cloud GPU training** (recommended): rent A100/H100 or use Colab/Kaggle/RunPod; full pretraining on 500M tokens.
2. **Parameter-efficient tuning locally**: load base weights in 4-bit (QLoRA) and train LoRA adapters on task data (CoT SFT, tool use). This does not change base weights but adds capability.
3. **CPU+RAM offload** with DeepSpeed (Linux only, not macOS/MPS) — not available here.

## 3. Execution plan (phased)

### Phase 0 — data (1–2 days)
- Build 500M-token deduplicated FineWeb v2 corpus with parallel parquet download + process pool (target ~20–40 min build).
- Build 200K-record CoT SFT dataset (OpenThoughts/Numina/OpenR1).

### Phase 1 — local full training (1–2 weeks)
- Nano: 500M tokens full pretraining.
- Alpha: 50M tokens (local), rest on cloud.
- Beta: 20M tokens (local), rest on cloud.
- Core: 10M tokens with 8-bit AdamW + gradient checkpointing (validation only).

### Phase 2 — capability SFT (local, all tiers)
- CoT SFT on Nano/Alpha/Beta full checkpoints.
- Tool-use SFT on Nano/Alpha (function-calling format).
- Evaluate on GSM8K, MATH-500, HumanEval, BFCL.

### Phase 3 — cloud full training (Core/Pro/Flagship)
- Provision GPU instance; upload corpus + checkpoints.
- Train Core (1.1B) on 500M tokens (~$50–150 on A100 spot).
- Train Pro (3.3B) on 500M tokens (~$200–600).
- Train Flagship (5.8B) on 500M tokens (~$400–1,200).
- Distill Flagship → Pro → Core → Beta → Alpha → Nano.

### Phase 4 — export & integration
- Export real GGUF for all tiers; import into Ollama with verified tags.
- Wire OpenCode provider; test tool calling end-to-end.

## 4. Immediate next commands

```bash
# Verify current datasets
.venv/bin/python scripts/verify_pretrain_shards.py --dataset-dir "/Volumes/One Touch/Helios/datasets/helios_fineweb_v1"

# Commit CoT builder
git add scripts/build_cot_dataset.py reports/fineweb_v1_shard_audit.json
git commit -m "feat(cot): add deduplicated CoT dataset builder"
git push origin pretrained
```

## 5. Key risks
- MPS training can be unstable with bf16 on some ops; fall back to fp32 for NaN steps.
- 8-bit AdamW on MPS may require CPU fallback; benchmark first.
- 500M-token training on M3 is days/weeks; set realistic expectations or use cloud.
- Never publish block-slice checkpoints as full models.

## 6. Definition of done
- All six tiers have full (non-slice) checkpoints with passing tensor/SHA/NaN gates.
- CoT and tool-use benchmarks measured and reported.
- Real GGUF + Ollama + OpenCode integration verified.
- All artifacts pushed to repo and (after HF auth) to Hub.
