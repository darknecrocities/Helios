# Teacher Selection & Distillation Plan

## Goal
Pick the best open teacher to distill into Helios tiers (Nano/Alpha/Beta/Core) on a 16 GB M3, using only the One Touch drive, and produce a concrete execution plan.

## Constraints
- Student vocab: **32,000** (Helios BPE). Teacher must share this vocab OR we add a vocab projection (costly).
- Teacher must run inference on 16 GB M3 (4-bit/8-bit) while student trains.
- Teacher should be strong at **coding + tool use** (user priority).
- Distillation quality scales with teacher strength and data volume.

## Candidate teachers (32k-vocab, coding-capable)

| Teacher | Params | Vocab | Coding strength | Fits 16GB (4-bit) | Notes |
|---|---:|---:|---|---|---|
| TinyLlama-1.1B-Chat | 1.1B | 32k | Weak | Yes | Fast but poor teacher |
| CodeLlama-7B-Instruct | 7B | 32k | Strong | Tight (~5GB 4-bit) | Best coding 32k teacher |
| Mistral-7B-v0.3 | 7B | 32k | Strong (general) | Tight | Good all-rounder |
| DeepSeek-Coder-6.7B | 6.7B | 32k? | Very strong code | Tight | Check vocab (32000) |
| Phi-3-mini-4k | 3.8B | 32k | Moderate | Yes | Fast, decent |
| Qwen2.5-Coder-1.5B | 1.5B | 151k | Strong | Yes | Vocab mismatch (needs projection) |

## Recommendation
**Primary teacher: CodeLlama-7B-Instruct** (32k vocab, coding-specialized, fits in 4-bit on 16GB).

- Run teacher in **4-bit (bitsandbytes / mlx)** for inference only.
- Student trains in fp16/bf16 with autocast.
- Use **logit distillation (KL)** + **CE on hard labels** + optional **feature distillation** (hidden-state MSE) for Nano/Alpha.

**Fallback if 7B is too slow on M3:** Phi-3-mini-4k (3.8B, fast, decent code) or Mistral-7B.

**Do NOT use** Qwen2.5-Coder as teacher directly (vocab mismatch) unless we add a linear vocab projection layer (adds complexity, slower).

## Feasible tiers for distillation on 16GB M3
| Tier | Teacher size | Student size | Mode | Est. time (50k samples) |
|---|---|---|---|---|
| Nano | 7B 4-bit | 17.8M | Full KD | ~6–10 h |
| Alpha | 7B 4-bit | 100M | Full KD | ~20–30 h |
| Beta | 7B 4-bit | 303M | KD + grad ckpt | ~60–90 h |
| Core | 7B 4-bit | 1.1B | QLoRA adapter KD | ~40–60 h |
| Pro/Flagship | — | — | Not feasible locally; use larger teacher on cloud |

## Plan (execute when you say "resume")
1. Install `bitsandbytes` (CPU/MPS fallback) or use `mlx-lm` for 4-bit teacher inference on Mac.
2. Update `scripts/distill_from_teacher.py`:
   - Default teacher → CodeLlama-7B-Instruct.
   - Add 4-bit teacher loading (`load_in_4bit=True`).
   - Add `--checkpoint-every` (already done) and `--vocab-project` flag (optional).
   - Add hidden-state MSE loss (feature distillation) for Nano/Alpha.
3. Run Nano distillation first (50k samples, ~6–10 h).
4. Evaluate on GSM8K / HumanEval before scaling to Alpha.
5. If Nano improves, run Alpha (50k) and Beta (20k).
6. Core: QLoRA adapter distillation only.

## Storage
- Teacher weights cached on One Touch (`HF_HOME` already symlinked).
- Student checkpoints on One Touch.
- Delete teacher cache after distillation if space is tight.

## Non-claims
- Distillation improves but does not guarantee "strong" coding; benchmark before claiming.
- 7B teacher on 16GB M3 is slow; expect long runs.
- Core/Pro/Flagship full distillation needs cloud GPUs.
