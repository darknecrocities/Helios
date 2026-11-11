# Helios v0.4.0 — 1M Context, Coding, Tools/MCP, and Vision

**Status:** research + implementation plan, with a verified conflict-free continuation lineage  
**Date:** 2026-11-11  
**Author:** Arron Kian Parejas `<parejasarronkian@gmail.com>`  
**Audit command:** `python scripts/audit_v040_capabilities.py --live --json-out reports/v040_capability_audit.json`

---

## 1. Executive decision

The requested end state is reachable only in stages. The current artifacts do **not** contain a full Core, Pro, or Flagship model, and they were trained on 96-token sequences. Metadata cannot turn a 96-token single-block slice into a 1M-context, tool-using, image-understanding model.

| Request | Current status | Why |
|---|---|---|
| Pro / Flagship ~1M context | **Target defined, not trained** | KV cache alone needs ~131 GB FP16 for Pro/Flagship; this host has 16 GB |
| Resume from latest checkpoint without conflict | **Done** | New `v040_lineage`; parents SHA-verified; Pro uses step 5145, not the mixed `latest` |
| Coding capability | **Pipeline validated, capability not proven** | 60 s / 256-token continuation on ~23M coding+agentic tokens; no code benchmark |
| Native tools | **Not present in Helios weights** | Pro tool call succeeds through the `llama3.2` base wrapper, not Helios |
| MCP | **Not implemented at runtime** | `mcp` package absent; no JSON-RPC client; tokenizer/README mentions are not a client |
| Image input | **Not present in Helios weights** | Flagship image probe succeeds through `gemma3:4b`; no trained Helios projector or `mmproj` |
| Ollama / HF sync | **Blocked by full weights** | Only block slices exist; publishing them as full tiers would be false advertising |

**Recommended product split**

1. **Local (16 GB M3):** 32K raw context by default, 128K for Core with `q8_0` KV, and HMA retrieval for 1M-scale document workflows. Do not advertise native 1M.
2. **Cloud:** native 1M only with full weights, long-context training, paged/quantized KV, and sequence parallelism on 8×H100/H200-class GPUs.
3. **Tools/MCP:** a runtime client plus tool-call SFT. Ollama's `tools` flag is API plumbing, not a trained capability.
4. **Vision:** a trained projector and image-text SFT, exported as an `mmproj`/adapter. A base-model vision flag is not a Helios capability.
5. **Release:** gate Ollama and HF publication on full-weight integrity, long-context evals, code evals, tool-schema validity, and vision evals.

---

## 2. Audited evidence

### 2.1 Host

| Item | Value |
|---|---|
| Machine | `Mac15,12` (Apple M3 MacBook Air) |
| CPU | 8 cores |
| Unified memory | 16.0 GiB |
| Internal free disk | ~67 GiB |
| One Touch free disk | ~642 GiB |
| PyTorch | 2.15.0, MPS available, CUDA unavailable |

This is sufficient for block-slice continuation and retrieval, not for full-parameter 3.3B/5.8B training or native 1M serving.

### 2.2 Checkpoints on the One Touch drive

Two different artifacts are stored under the same tier names: **full models** (Nano, Alpha, Beta) and **single-block slices** (Core, Pro, Flagship).

| Tier | Advertised unique params | Stored state-dict params | Tensors | Full model? | Effective step | Mode |
|---|---:|---:|---:|---|---:|---|
| Nano | 17,842,880 | 28,082,880 | 66/66 | yes | 17,465 | full |
| Alpha | 100,092,672 | 124,668,672 | 111/111 | yes | 4,820 | full |
| Beta | 303,350,784 | 336,118,784 | 219/219 | yes | 2,209 | full |
| Core | 1,128,892,416 | 52,703,744 | 12/219 | no | 11,304 | block slice |
| Pro | 3,319,729,152 | 113,265,152 | 12/291 | no | 5,145 | block slice |
| Flagship | 5,802,037,248 | 194,000,896 | 12/291 | no | 6,119 | block slice |

The authoritative audit is `scripts/audit_full_family_parameters.py`; its output is `reports/full_family_parameters.json`.

Important details:

- **Advertised size is the unique parameter count.** Tied embeddings are one storage block, but `model.state_dict()` returns both `embed_tokens.weight` and `lm_head.weight`, which alias the same storage. That is why stored state-dict totals are larger (for example, Nano 17.84M unique vs 28.08M stored). The checkpoint is not “bigger than advertised”; it contains a second reference to the same tied embedding.
- The full-model gate is: no `block.*` keys, `9 × n_layers + 3` tensors, and all tensor indices `layers.0 … layers.n_layers-1` present. Expected counts are 66, 111, 219, 219, 291, 291 for Nano, Alpha, Beta, Core, Pro, Flagship.
- A block-slice checkpoint contains **one** transformer block plus embeddings, norm, and LM head. Core should contain 24 blocks; Pro and Flagship should contain 32.
- `scripts/pretrain_all_models.py` implements `ProgressiveTierProjectionHead`, which is a memory-safe slice. It is useful for pipeline experiments; it is not a full model.
- `helios/training/controller.py::adapt_block_streamed_weights` reconstructs a full model by **replicating the same trained block across every layer**. That produces a degenerate model with tied layer behavior. It must not be published as a trained tier.
- Pro's `latest` is a mixed copy:
  - `state.pt` is from step 5,053.
  - `model.safetensors` is from step 5,145.
  - The v0.4.0 lineage therefore uses `pro_continuous/step_00005145`, which is internally consistent.
- Nano, Alpha, and Beta are the only real full models on the drive. They are small and were trained at very short sequence lengths (`max_seq_len` 160/160/128), so they are valid local training parents but not substitutes for Core/Pro/Flagship.
- Core/Pro/Flagship `config.json` says `max_seq_len: 64`, while manifests record `seq=96`. Both are far below 128K, let alone 1M.

**Consequence for distillation:** a teacher must pass the same full-model gate. Flagship is currently 12/291 tensors, so it cannot teach Pro, Core, Beta, Alpha, or Nano. `scripts/distill_family_lineage.py` enforces this and prints `TEACHER GATE FAILED` instead of silently distilling from a slice.

### 2.3 What the corpus actually contains

`/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl`:

| Domain | Records | Text chars | Estimated tokens |
|---|---:|---:|---:|
| Coding and algorithms | 35,020 | 43.4M | 11.1M |
| Software engineering and architecture | 18,619 | 13.0M | 3.3M |
| Tech history / coding documentation | 10,000 | 8.6M | 2.2M |
| Agentic skills and MCP | 10,000 | 23.5M | 6.0M |
| All domains | 134,728 | 133.1M | **~34.0M** |

Token estimates use a measured 3.91 chars/token sample. The JSONL file is 402 MB on disk, but most of that is JSON structure and metadata; the actual text is ~133M characters.

This matters:

- Coding + software engineering + documentation + agentic/MCP is roughly **22.6M tokens**.
- A Chinchilla-style 20 tokens/parameter budget is ~22.6B tokens for Core, ~66.4B for Pro, and ~116B for Flagship.
- The entire corpus is therefore **0.03%–0.15%** of a compute-optimal pretraining budget for these model sizes.
- A full epoch over the corpus cannot produce a competitive 3.3B/5.8B model, regardless of hardware.

The HMA FTS5 index at `/Volumes/One Touch/Helios/data/hma_full_corpus.db` contains **134,728 documents** and is a real retrieval asset. It is not a substitute for native long-context training.

### 2.4 Ollama capability audit

| Tag | Served through | Context header | Advertised capabilities | Live result |
|---|---|---:|---|---|
| `parejasarronkian/Helios:core` | `gemma2:2b` | 131,072 | `completion` | text-only |
| `parejasarronkian/Helios:pro` | `llama3.2:latest` | 131,072 | `tools`, `completion` | tool call verified: `get_weather({"city":"Manila"})` |
| `parejasarronkian/Helios:flagship` | `gemma3:4b` | 262,144 | `completion`, `vision` | image accepted; response generated |

Interpretation:

- The tags are **base models with Helios system prompts and parameters**, not Helios weights.
- Pro's tool call is a genuine runtime test of the `llama3.2` base model. It does not verify Helios-native tool use.
- Flagship's image test is a genuine runtime test of `gemma3:4b` vision. It does not verify a Helios vision adapter.
- Core's template may mention tools, but `capabilities` is only `completion`; it refused an actual tool call in earlier testing.
- `ollama show` reports `context_length` from GGUF metadata. It is a serving header, not evidence that the weights were trained at that length.

### 2.5 Hugging Face publication audit

Repo: `Darknecrocities/Helios` (public), main SHA `5c9b60cc5c76...`, last modified 2026-11-10.

| Tier branch | Hub continuous step | Drive effective step | State |
|---|---:|---:|---|
| `core` | 9,210 | 11,304 | stale |
| `pro` | 4,142 | 5,145 | stale |
| `flagship` | 2,390 | 6,119 | stale |

The Hub `main` branch also contains `._*` AppleDouble files and sparse/zero-filled tier safetensors. The tier branches have architecture `config.json` files but not verified full trained weights. Do not treat the Hub as a source of truth for capability.

### 2.6 Tokenizer and model code that already exists

- `<|call:tool|>`, `<|return:tool|>`, `<|image|>`, `<|memory_start|>`, `<|memory_end|>`, and refusal tokens are defined in `helios/tokenizer/special_tokens.py`.
- The chat template knows how to render tool calls, tool returns, images, and HMA memory context.
- `helios/model/multimodal.py` contains a ViT-style encoder, a GELU projector, and a multimodal attention mask.
- `helios/export/hf.py` writes a `vision_config` block when `d_model >= 2048`.
- `helios/model/rotary_embedding.py` implements dynamic NTK-aware RoPE scaling beyond the nominal `max_seq_len`.

These are **scaffolding**, not trained capabilities. The missing pieces are full weights, long-context training, tool-call SFT, a vision projector checkpoint + `mmproj` export, and an MCP runtime.

---

## 3. What “1M context” actually costs

A context window is three separate constraints:

1. **Trained positional distribution** — the model must have seen long sequences and learned to use them.
2. **KV cache memory** — every layer stores keys and values for every token.
3. **Serving compute and scheduling** — dense attention is quadratic; memory bandwidth limits decode.

### 3.1 KV cache at 1M tokens (K + V, not one tensor)

Formula:

```
KV bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_element
head_dim = d_model / n_heads
```

All three tiers use `head_dim = 128`.

| Tier | KV FP16 @ 32K | KV FP16 @ 128K | KV FP16 @ 256K | KV FP16 @ 1M | KV INT8 @ 1M | KV Q4 @ 1M |
|---|---:|---:|---:|---:|---:|---:|
| Core | 1.61 GB | 6.44 GB | 12.88 GB | **49.15 GB** | 24.58 GB | 12.29 GB |
| Pro | 4.29 GB | 17.18 GB | 34.36 GB | **131.07 GB** | 65.54 GB | 32.77 GB |
| Flagship | 4.29 GB | 17.18 GB | 34.36 GB | **131.07 GB** | 65.54 GB | 32.77 GB |

Weights in FP16: Core 2.26 GB, Pro 6.64 GB, Flagship 11.60 GB.

| Tier | Total FP16 KV + weights @ 1M | Total INT8 KV + FP16 weights @ 1M |
|---|---:|---:|
| Core | 51.4 GB | 26.8 GB |
| Pro | 137.7 GB | 72.2 GB |
| Flagship | 142.7 GB | 77.1 GB |

These totals exclude activations, attention workspace, allocator fragmentation, and the OS. A 16 GB laptop cannot hold even Pro's 128K FP16 KV cache plus weights (17.18 + 6.64 = 23.8 GB). With INT8 KV, Pro at 128K is ~15.2 GB, which is already at the edge of a 16 GB machine before overhead.

### 3.2 Attention compute and bandwidth at 1M

For one query token attending to 1M keys/values, dense attention costs approximately:

```
4 × seq_len × d_model × n_layers FLOPs
```

| Tier | Attention FLOPs per decoded token @ 1M | KV bytes read per token @ 1M |
|---|---:|---:|
| Core | 196.6 PFLOPs | 49.15 MB |
| Pro | 393.2 PFLOPs | 131.07 MB |
| Flagship | 524.3 PFLOPs | 131.07 MB |

At an optimistic 400 TFLOP/s effective per H100, one decoded token is ~0.5–1.3 s of pure attention compute. Decode is also memory-bound: at ~3.35 TB/s HBM bandwidth, reading 131 MB of KV takes ~39 ms, a theoretical ceiling of ~25 tokens/s for Pro/Flagship at 1M (before overhead). On the M3's ~100 GB/s unified memory, 131 MB/token is ~1.3 s/token just for KV reads.

Prefill is far worse: a full 1M-token prefill is roughly `seq_len` times the per-token attention cost, i.e. hundreds of zettaFLOPs. Even with causal masking, FlashAttention, and 8 GPUs, native 1M prefill is a multi-hour, multi-GPU operation. This is why production 1M-context systems use paged attention, prefix caching, sequence parallelism, and often sparse/ring attention.

### 3.3 Training memory

Full fine-tuning with AdamW in mixed precision needs roughly:

```
FP16 weights + FP16 gradients + FP32 Adam m/v (+ optional FP32 master weights)
```

| Tier | Weights + grads + Adam m/v | With FP32 master | Minimum practical cluster |
|---|---:|---:|---|
| Core | ~9.0 GB | ~13.6 GB | 1×24–48 GB for short context; multi-GPU for 1M |
| Pro | ~39.8 GB | ~53.1 GB | 8×80 GB with FSDP/ZeRO-3 + activation checkpointing |
| Flagship | ~69.6 GB | ~92.8 GB | 8×80 GB minimum, 8×H200 141 GB preferred |

At 1M tokens, activation memory dominates. A single layer's input at 1M tokens is `1e6 × d_model × 2 bytes`: 4.1 GB for Core, 6.1 GB for Pro, 8.2 GB for Flagship. Storing every layer's activations is impossible; training requires activation checkpointing, sequence parallelism, and sharding. This is not a laptop workload.

---

## 4. What was actually trained in v0.4.0

### 4.1 Command

```bash
.venv/bin/python scripts/continue_v040_lineage.py \
  --seconds 60 --seq-len 256 --batch-size 1 --lr 5e-5 \
  --data-limit 9000 --eval-limit 300
```

### 4.2 Parents and integrity

| Tier | Parent | Parent step | Parent SHA-256 (prefix) |
|---|---|---:|---|
| Core | `core_continuous/latest` | 11,304 | `9b7037a78d4a25a5` |
| Pro | `pro_continuous/step_00005145` | 5,145 | `16779fd14519a445` |
| Flagship | `flagship_continuous/latest` | 6,119 | `c1d0f24f9b2cbcea` |

The runner verifies the parent safetensors SHA against the parent manifest, rejects NaN/Inf parameters, writes to a staging directory, and moves only completed checkpoints into `v040_lineage`. It never writes to `<tier>_continuous/latest`.

### 4.3 Results

| Tier | New step | Steps run | Tokens seen | Held-out before | Held-out after | Relative improvement |
|---|---:|---:|---:|---:|---:|---:|
| Core | 11,842 | 538 | 56,041 | 9.966 | 7.021 | 29.6% |
| Pro | 5,369 | 224 | 22,947 | 13.954 | 7.528 | 46.1% |
| Flagship | 6,234 | 115 | 11,747 | 12.082 | 7.923 | 34.4% |

Output:

```
/Volumes/One Touch/Helios/checkpoints/v040_lineage/
  core_continuous/step_00011842/
  pro_continuous/step_00005369/
  flagship_continuous/step_00006234/
  lineage.json
```

**Interpretation, stated conservatively:**

- The resume path, SHA verification, coding/agentic data selection, held-out loss measurement, and atomic save all work.
- The held-out split is drawn from the same coding/agentic domains, so the loss drop shows adaptation to that split. It is not a code-generation benchmark, a tool-call benchmark, or evidence of 1M context.
- 256 tokens per sample and 60 seconds per tier are pipeline validation settings, not a capability run. The models were previously trained at 96 tokens; 256 is still three orders of magnitude below 128K and four below 1M.
- The tokens seen (11.7K–56K) are a tiny fraction of the ~34M-token corpus and a vanishing fraction of a compute-optimal pretraining budget.

Do not publish these lineage checkpoints as “1M-capable” or “coding models.” Publish them, if at all, as **pipeline validation checkpoints** with the manifest’s `context_extension_status: target_only_not_trained`.

### 4.4 v0.5.0 full-parameter lineage (Nano, Alpha, Beta)

`scripts/continue_full_family_lineage.py` trains only tiers with a verified **full** parent. Core, Pro, and Flagship are refused because their on-drive checkpoints are 12-tensor block slices.

```bash
.venv/bin/python scripts/continue_full_family_lineage.py \
  --tiers nano,alpha,beta --seconds 60 --seq-len 256 --batch-size 1 \
  --lr 5e-5 --data-limit 9000 --eval-limit 300
```

| Tier | Parent step | New step | Steps run | Tokens seen | Tensors | Unique params | Held-out before | Held-out after | Relative improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Nano | 17,465 | 18,577 | 1,112 | 116,419 | 66/66 | 17,842,880 | 6.706638 | 5.712176 | 14.8% |
| Alpha | 4,820 | 5,010 | 190 | 19,704 | 111/111 | 100,092,672 | 6.717248 | 6.098469 | 9.2% |
| Beta | 2,209 | 2,260 | 51 | 5,332 | 219/219 | 303,350,784 | 7.509425 | 6.861714 | 8.6% |

Verification performed on every output:

- No `block.*` keys.
- Layer indices present for all 7/12/24 blocks.
- Stored tensor sums match the tied-embedding formula (28,082,880 / 124,668,672 / 336,118,784).
- `model.safetensors` SHA-256 matches the manifest.
- NaN/Inf scan passes before save.
- Writes only to `v050_full_lineage`, never to the parent directories.

**This is still not pretraining.** It is a 60-second continued-training run per tier at 256 tokens. Nano saw 116K tokens; Alpha 19.7K; Beta 5.3K. The held-out split is coding/agentic text from the same local seed corpus, so the loss drop is adaptation to that split, not evidence of FineWeb-scale pretraining or chain-of-thought capability.

Output:

```
/Volumes/One Touch/Helios/checkpoints/v050_full_lineage/
  nano_full/step_00018577/
  alpha_full/step_00005010/
  beta_full/step_00002260/
  lineage.json
```

---

## 5. Capability requirements and gaps

### 5.1 Coding

**What is needed**

- A full model, not a replicated block slice.
- Continued pretraining on a much larger code corpus (the current ~16.6M code/doc tokens is a seed, not a pretraining set).
- Repository-level data: files, diffs, tests, issues, build logs, and execution feedback.
- SFT for code completion, bug fixing, refactoring, and test generation.
- Preference/RL tuning using unit-test pass/fail as the reward.
- Sandboxed evaluation with pass@1, not loss alone.

**Current gap**

The v0.4.0 run used 9,000 training records at 256 tokens and improved held-out loss. That validates the data path. It does not demonstrate HumanEval/MBPP/SWE-bench capability. The 35,020 coding records are mostly instructional text, not executable repository tasks.

**Minimum eval gate before claiming coding**

- HumanEval+ / MBPP+ pass@1 in a sandbox.
- A held-out repository-edit set with tests.
- No test-set contamination; record dataset hashes and deduplication.
- Compare against the parent checkpoint on the same prompts, not only against itself.

### 5.2 Tools

**What already exists**

- Tool-call tokens in the tokenizer.
- A chat template that can render tool definitions and tool returns.
- A procedural memory registry with JSON-Schema validation in `helios/memory/procedural.py`.
- Ollama's `tools` API for the Pro tag.

**What is missing**

- Helios-native tool-call SFT. The current Pro tag calls tools because it is `llama3.2`, not because Helios learned to.
- A strict parser for `<|call:tool|>name(json)<|return:tool|>` with schema validation, retries, and error feedback.
- A runtime executor with allowlists, timeouts, idempotency keys, and audit logs.
- Constrained/grammar-based decoding if the model emits malformed JSON.
- Evaluation on BFCL-style tool selection, argument validity, and multi-step execution.

**Ollama truth**

`ollama show` advertises `tools` when the template references `.Tools`. That is a serving contract. It does not make a model good at tool use. Core advertises nothing; Pro advertises tools and passes a live call through its base model; Flagship advertises vision and accepts an image through `gemma3`.

### 5.3 MCP

MCP is a **client/server protocol**, not a model weight. A compliant implementation needs:

- JSON-RPC 2.0 framing over stdio or HTTP/SSE.
- `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, and error handling.
- Server registry, capability negotiation, authentication, timeouts, and cancellation.
- Tool-schema injection into the model prompt and parsing of model tool calls.
- Sandboxing and an allowlist for filesystem/network/process access.
- Audit logging and prompt-injection defenses for tool outputs and retrieved resources.

**Current gap**

The `mcp` Python package is not installed. There is no `helios/mcp` client module. References in README/landing-page copy and synthetic MCP training records are not a runtime. The audit reports `mcp: runtime_missing`.

**Recommended design**

```
User → Helios runtime → MCP client manager → MCP servers (stdio/HTTP)
                         ↕ tool schemas
                    Helios model prompt
                         ↕ tool calls/results
                    Executor + policy + audit
```

Ollama can be the model backend, but the MCP client must live in the Helios runtime (or an MCP host). Do not claim Ollama itself speaks MCP.

### 5.4 Image input

**What already exists**

- `<|image|>` token and chat-template rendering.
- Vision encoder, projector, and multimodal attention mask.
- HF export writes a `vision_config` block for `d_model >= 2048`.

**What is missing**

- A trained projector aligned to the Helios text embedding space.
- Image-text pretraining/SFT data and a training recipe.
- An exported vision encoder + projector (for Ollama, an `mmproj` GGUF or equivalent).
- Image preprocessing, resolution handling, and multimodal batching.
- Vision evals (captioning, VQA, OCR, chart reading) and safety evals.

**Current gap**

The Flagship tag accepts an image because it is `gemma3:4b`. No Helios `mmproj` exists in the repo or on the drive. The HF `vision_config` is a declaration, not a trained adapter.

### 5.5 Long context

**What already exists**

- Dynamic NTK-aware RoPE scaling in `RotaryEmbedding`.
- Presets with nominal 128K/256K contexts.
- GGUF header calibration and `num_ctx` parameters.
- HMA retrieval infrastructure over 134,728 documents.

**What is missing**

- Long-context training data and curriculum.
- Evaluation at 32K/128K/256K/1M (RULER, needle-in-a-haystack, LongBench, repo-level tasks).
- Paged/quantized KV serving.
- Sequence parallelism for training.
- A truthful separation between “trained context” and “configured context.”

**Current gap**

The checkpoints were trained at 96 tokens. The Ollama headers say 128K/256K. The audit reports `trained_96_header_131072` / `trained_96_header_262144`. That is the correct description.

---

## 6. Recommended plan

### Phase 0 — lineage and audit (completed)

- Verify parent SHA-256; reject mixed/corrupt checkpoints.
- Use Pro `step_00005145`, not `latest`.
- Write only to `v040_lineage`; never overwrite existing training.
- Add `scripts/continue_v040_lineage.py`.
- Add `scripts/audit_v040_capabilities.py` and a JSON report.
- Add target profiles `configs/models/{core-v040,pro-1m,flagship-1m}.yaml`.

### Phase 1 — obtain a real full model (blocking)

Pick one:

**A. Full pretraining from scratch (recommended for a real foundation model)**

- Data: trillions of tokens, not the current 34M-token corpus.
- Compute: 8×H100/H200 for weeks, depending on token budget.
- Architecture: keep the Helios decoder, GQA, RoPE, SwiGLU, Pre-RMSNorm.
- Checkpoint format: full `HeliosForCausalLM` state dict, 219 tensors (Core) or 291 (Pro/Flagship).
- Gate: full-weight tensor count, SHA manifest, loss curve, held-out perplexity, and a reproducible data manifest.

**B. Recover or retrain the missing full checkpoints**

- Search all drives/backups for full `layers.*` checkpoints. The current drive has only block slices for Core/Pro/Flagship.
- If a full checkpoint exists, verify SHA, tensor count, config, and optimizer state before resuming.
- Do not use `adapt_block_streamed_weights` for a release; it replicates one block across all layers.

**C. Distill from strong teachers into the Helios architecture**

- Expensive and legally sensitive; teacher licenses and output terms must be checked.
- Distillation can bootstrap capability but is not a substitute for a clean pretraining data pipeline.

**Decision gate:** do not start long-context or tool SFT until a full model passes the tensor-count and integrity gate.

#### Distillation order (Flagship-first)

`scripts/distill_family_lineage.py` implements the intended capacity cascade. It refuses to start unless the teacher passes the full-model gate.

| Stage | Teacher | Student | Student unique params | Teacher + student + AdamW | Minimum GPU |
|---:|---|---|---:|---:|---:|
| 1 | Flagship 5.80B | Pro 3.32B | 3,319,729,152 | 48.52 GiB | 80 GB |
| 2 | Pro 3.32B | Core 1.13B | 1,128,892,416 | 19.23 GiB | 24 GB |
| 3 | Core 1.13B | Beta 303M | 303,350,784 | 5.73 GiB | 24 GB |
| 4 | Beta 303M | Alpha 100M | 100,092,672 | 1.84 GiB | 24 GB |
| 5 | Alpha 100M | Nano 17.84M | 17,842,880 | 0.47 GiB | 24 GB |

Direct Flagship → any student is also supported. The gate currently fails:

```
TEACHER GATE FAILED: flagship at /Volumes/One Touch/Helios/checkpoints/helios_flagship
is a block slice (12 tensors). A distillation teacher must be a full model with
291 tensors and no block.* keys.
```

So the Flagship-first order is policy and code, not a completed run. Stage 1 becomes possible only after a full 291-tensor Flagship checkpoint exists.

### Phase 2 — context extension curriculum

Train in stages, evaluating at each stage:

| Stage | Raw context | Purpose | Serving |
|---|---:|---|---|
| 1 | 4K–8K | stable short-context baseline | local |
| 2 | 32K | local long-document work | 16 GB laptop with q8 KV |
| 3 | 128K | Core target; Pro/Flagship cloud | 24–48 GB GPU |
| 4 | 256K | Flagship target | 80 GB GPU with paged KV |
| 5 | 1M | cloud-only native target | 8×H100/H200, paged/quantized KV |

Technical choices:

- Use YaRN or LongRoPE for frequency scaling; keep dynamic NTK as a fallback.
- Adjust `rope_theta` and `max_seq_len` together; do not only patch the GGUF header.
- Pack sequences to reduce padding; use document-level attention masks where needed.
- Use activation checkpointing, FSDP/ZeRO-3, sequence parallelism, and bf16 (not fp16) for stability.
- Data: long code repositories, books, documentation, multi-file diffs, and synthetic long-context tasks. The current corpus has no meaningful 128K/1M documents.
- Budget: billions of tokens across the long stages; the exact amount depends on the base model and target evals. Treat “a few thousand tokens” or “one epoch of 34M tokens” as a smoke test, not context extension.

Evaluation:

- Needle-in-a-haystack at 32K/128K/256K/1M.
- RULER (retrieval, multi-hop, aggregation, QA).
- LongBench / LongBench-Chat.
- Repository-level code tasks with tests.
- “Lost in the middle” position sweeps.
- Degradation checks on short-context benchmarks after extension.

### Phase 3 — coding capability

1. **Data**
   - Expand beyond 16.6M code/doc tokens.
   - Deduplicate against eval sets; track licenses and provenance.
   - Add repository snapshots, diffs, tests, issues, and execution traces.
2. **Continued pretraining**
   - Code-heavy mix with replay of general text to avoid catastrophic forgetting.
   - Train at 4K–32K first; code repositories benefit from longer contexts.
3. **SFT**
   - Code completion, bug fixing, refactoring, test generation, and explanation.
   - Include tool use: read file, search, run tests, apply patch.
4. **Preference/RL**
   - Reward = unit tests pass, lint/type checks pass, patch applies, no regressions.
   - Reject reward hacking; run tests in a sandbox.
5. **Evaluation**
   - HumanEval+, MBPP+, LiveCodeBench, SWE-bench Lite/Verified, RepoBench.
   - Report pass@1 with confidence intervals and contamination checks.
   - Compare parent vs new checkpoint on identical prompts.

### Phase 4 — tools and MCP

1. **Tool-call SFT**
   - Data: query → tool selection → JSON arguments → tool result → final answer.
   - Include malformed calls and recovery; teach the model to ask for missing arguments.
   - Use the existing `<|call:tool|>` / `<|return:tool|>` tokens.
2. **Parser and executor**
   - JSON-Schema validation; reject unknown tools/arguments.
   - Retry with error feedback; cap retries.
   - Timeouts, allowlists, sandboxing, audit logs.
3. **MCP client**
   - Implement `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`.
   - Support stdio and HTTP/SSE transports.
   - Add server registry and per-server permissions.
   - Treat tool/resource content as untrusted; sanitize before injecting into the prompt.
4. **Ollama integration**
   - Use Ollama's `tools` API for the model boundary.
   - Keep MCP orchestration in the Helios runtime, not in the Modelfile.
5. **Evaluation**
   - BFCL-style tool selection and argument validity.
   - Multi-step tasks with real MCP servers.
   - Prompt-injection and exfiltration tests.

### Phase 5 — vision

1. **Data**
   - Image-text pairs, captions, VQA, OCR, charts, screenshots, and code screenshots.
   - Safety and bias evaluation sets.
2. **Training**
   - Start with a frozen vision encoder and train the projector; then selectively unfreeze.
   - Align image tokens with the text embedding space; use the existing multimodal attention mask.
3. **Export**
   - Save the vision encoder + projector separately.
   - Export an `mmproj`/adapter artifact for Ollama; document the exact Ollama version and flags.
   - Add image preprocessing and token budgeting to the runtime.
4. **Evaluation**
   - Captioning, VQA, OCR, chart/table reading, screenshot-to-code.
   - Refusal and safety behavior on sensitive images.

### Phase 6 — release gates

Do not publish to Ollama/HF until all gates pass:

- [ ] Full-model tensor count: 219 (Core) / 291 (Pro/Flagship), no `block.*` keys.
- [ ] SHA-256 manifest matches; no mixed `latest`.
- [ ] Config and tokenizer hashes recorded.
- [ ] Long-context evals at the advertised length.
- [ ] Code evals with a sandbox and contamination report.
- [ ] Tool-call schema validity and execution success.
- [ ] MCP integration tests against at least two servers.
- [ ] Vision evals with a trained adapter, not a base-model flag.
- [ ] Model card states trained context vs configured context.
- [ ] Ollama `num_ctx` does not exceed trained/served context.
- [ ] HF weights are real, not sparse/zero-filled placeholders.

---

## 7. Serving plan

### 7.1 Local 16 GB M3

Recommended defaults:

| Tier | Raw context | KV type | Notes |
|---|---:|---|---|
| Core | 32K default; 128K max | `q8_0` | 128K needs ~5.5 GB total with INT8 KV + weights; leave headroom |
| Pro | 32K | `q8_0` | 128K INT8 KV + weights is ~15.2 GB before overhead; not safe |
| Flagship | 16K–32K | `q8_0` or `q4_0` | 128K is ~20.2 GB with INT8 KV + weights; not safe |

Ollama environment:

```bash
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1
```

Keep `num_ctx` truthful. A 32K setting on a 96-token-trained model is still a 32K *configured* window; it is not a 32K *trained* capability.

### 7.2 1M-scale retrieval workflow (available now)

For large documents, use HMA rather than native 1M attention:

1. Index documents in the FTS5 corpus (`corpus_fts`, 134,728 docs).
2. Retrieve top-k with BM25 + dense + reciprocal rank fusion.
3. Sanitize retrieved content for prompt injection.
4. Assemble a bounded evidence block (`build_grounded_context`).
5. Answer inside a 32K/128K raw window.

This supports “answer questions over a 1M-token library,” but it is **not** equivalent to native 1M attention. It can miss evidence that requires reasoning over the entire document, and retrieval recall becomes the bottleneck. Evaluate retrieval recall separately from generation quality.

### 7.3 Cloud native 1M

Recommended stack:

- vLLM, SGLang, or TensorRT-LLM with paged attention and prefix caching.
- 8×H100 80GB or 8×H200 141GB.
- INT8 KV cache (`q8_0`) at minimum; FP16 KV only on 192GB+ nodes.
- Sequence parallelism / ring attention for prefill.
- Chunked prefill and request scheduling to avoid monolithic 1M prefills.
- Cost controls: cap `max_model_len`, use prefix caching, and route long requests to a separate pool.

Rough planning numbers (not a quote):

- Pro FP16 KV at 1M: ~137.7 GB total; 192 GB node recommended.
- Pro INT8 KV at 1M: ~72.2 GB total; 96 GB node minimum, 128 GB comfortable.
- Flagship FP16 KV at 1M: ~142.7 GB total; 192 GB node recommended.
- Flagship INT8 KV at 1M: ~77.1 GB total; 128 GB node recommended.

### 7.4 Ollama policy

- Keep the existing v0.3.0 tags as **base-model wrappers** until full Helios weights exist.
- Do not add `.Tools` or vision flags to a tag unless the underlying model and evals support them.
- When a full model is ready, build from a local GGUF (`FROM /path/helios-core-v040.gguf`) rather than a base model.
- For vision, ship the `mmproj` and document the Ollama version; test image input end to end.
- For tools, use Ollama's `tools` API and keep MCP in the runtime.

### 7.5 Hugging Face policy

- Publish only full, verified safetensors with a matching SHA manifest.
- Use a separate branch or folder for v0.4.0; do not overwrite v0.3.0.
- Include `config.json`, tokenizer files, generation config, model card, eval report, and data provenance.
- Mark block-slice or lineage-demo checkpoints explicitly; do not label them as full tiers.
- Remove or quarantine `._*` AppleDouble files and sparse placeholders from release branches.

---

## 8. Verification commands

```bash
# Capability matrix (add --live to probe tools/vision)
.venv/bin/python scripts/audit_v040_capabilities.py \
  --live --json-out reports/v040_capability_audit.json

# Resume dry run (no writes)
.venv/bin/python scripts/continue_v040_lineage.py \
  --dry-run --seq-len 256 --data-limit 9000 --eval-limit 300

# Bounded lineage continuation (writes only to v040_lineage)
.venv/bin/python scripts/continue_v040_lineage.py \
  --seconds 60 --seq-len 256 --batch-size 1 --lr 5e-5 \
  --data-limit 9000 --eval-limit 300

# Unit/integration tests
.venv/bin/pytest tests/ -q

# Ollama spot checks
ollama show parejasarronkian/Helios:core
ollama show parejasarronkian/Helios:pro
ollama show parejasarronkian/Helios:flagship
```

For a future full-weight release, add:

```bash
# Full-model integrity gate (to be implemented with the release script)
python scripts/verify_v040_release_gates.py \
  --checkpoint /path/to/full/checkpoint \
  --eval-report reports/v040_eval.json
```

---

## 9. Risks and non-claims

1. **Do not claim native 1M context.** Existing Core/Pro/Flagship slices were trained at 96 tokens; the Nano/Alpha/Beta parents at 160/160/128; the v0.5.0 full continuations at 256. The 128K/256K values in Ollama and configs are headers/targets.
2. **Do not claim Helios tool use from the Pro tag.** The verified tool call is `llama3.2`.
3. **Do not claim Helios vision from the Flagship tag.** The verified image input is `gemma3:4b`.
4. **Do not publish replicated block slices.** `adapt_block_streamed_weights` ties every layer to one trained block; it is a compatibility shim, not a trained model.
5. **Do not treat HMA retrieval as native attention.** It is a bounded-context workflow with retrieval recall risk.
6. **Do not train full models on the current corpus alone.** ~34M tokens is 0.03%–0.15% of a compute-optimal budget for these sizes.
7. **Do not ignore Pro's mixed `latest`.** It remains on disk. The lineage uses step 5145; the old `latest` should be repaired only after a backup and explicit approval.
8. **Watch numerical stability.** Use bf16 for long-context training; fp16 attention can overflow at long sequence lengths.
9. **Watch data licensing and contamination.** Code, books, and image data need provenance and eval-set deduplication.
10. **Watch thermal throttling.** A 16 GB M3 Air will throttle under sustained training; use it for smoke tests, not multi-day runs.

---

## 10. Immediate next actions

1. **Decide the full-model path:** full pretraining (A), recover full checkpoints (B), or distillation (C). This is the blocking decision.
2. **Approve a cloud budget** for full pretraining and long-context extension. Without it, 1M remains a target profile.
3. **Keep the v0.4.0 lineage as evidence**, not a release. It proves the resume and data pipeline.
4. **Implement the MCP client** and a tool executor before claiming MCP support.
5. **Build the coding eval harness** (sandbox + HumanEval+/MBPP+/repo tasks) before claiming coding gains.
6. **Train and export a vision projector** before enabling image input on any Helios tag.
7. **Publish only documentation/metadata now** if desired; do not push block slices as full models.
8. **Repair Pro `latest`** only after backing up the existing directory and confirming the desired step (5,145).

---

## Appendix A — Exact provenance

| Item | Path |
|---|---|
| Core parent | `/Volumes/One Touch/Helios/checkpoints/core_continuous/latest` |
| Pro parent | `/Volumes/One Touch/Helios/checkpoints/pro_continuous/step_00005145` |
| Flagship parent | `/Volumes/One Touch/Helios/checkpoints/flagship_continuous/latest` |
| New lineage | `/Volumes/One Touch/Helios/checkpoints/v040_lineage` |
| Full-family lineage | `/Volumes/One Touch/Helios/checkpoints/v050_full_lineage` |
| Distillation lineage | `/Volumes/One Touch/Helios/checkpoints/v060_distill_lineage` |
| Corpus | `/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl` |
| HMA FTS5 DB | `/Volumes/One Touch/Helios/data/hma_full_corpus.db` |
| Continuation runner | `scripts/continue_v040_lineage.py` |
| Full-model runner | `scripts/continue_full_family_lineage.py` |
| Distillation runner | `scripts/distill_family_lineage.py` |
| Parameter audit | `scripts/audit_full_family_parameters.py` → `reports/full_family_parameters.json` |
| Audit runner | `scripts/audit_v040_capabilities.py` |
| Audit JSON | `reports/v040_capability_audit.json` |
| Target profiles | `configs/models/core-v040.yaml`, `pro-1m.yaml`, `flagship-1m.yaml` |

## Appendix B — Full-model tensor-count gate

For `HeliosForCausalLM` with tied embeddings, the expected state-dict tensor count is:

```
Nano:     1 embed +  7 × 9 block tensors + 1 final norm + 1 lm_head =  66
Alpha:    1 embed + 12 × 9 block tensors + 1 final norm + 1 lm_head = 111
Beta:     1 embed + 24 × 9 block tensors + 1 final norm + 1 lm_head = 219
Core:     1 embed + 24 × 9 block tensors + 1 final norm + 1 lm_head = 219
Pro:      1 embed + 32 × 9 block tensors + 1 final norm + 1 lm_head = 291
Flagship: same as Pro                                                   = 291
```

A checkpoint with 12 tensors and keys beginning with `block.` is a slice. A checkpoint with `layers.0 ... layers.N-1` and the expected count is a full model candidate.

## Appendix C — KV formula and assumptions

```
head_dim = d_model / n_heads
KV elements per token = 2 × n_layers × n_kv_heads × head_dim
FP16 bytes = elements × 2
INT8 bytes = elements × 1
Q4 bytes = elements × 0.5
```

Assumptions: batch size 1, no tensor parallelism duplication, no prefix cache, no allocator overhead. Real serving needs 10–30% headroom.

## Appendix D — Glossary

- **Block slice:** one transformer block trained at the tier's hidden size; not a full model.
- **Full model:** all layers present, with expected tensor count and config.
- **Native context:** the sequence length the weights were trained on and can attend over.
- **Configured context:** the `num_ctx`/`max_seq_len` header value at serving time.
- **Effective context:** retrieval-augmented workflow that answers over a larger corpus inside a bounded window; not native attention.
- **MCP:** Model Context Protocol, a JSON-RPC client/server protocol for tools and resources.
- **mmproj:** multimodal projector artifact used by llama.cpp/Ollama-style vision models.
- **YaRN / LongRoPE:** RoPE frequency-scaling methods for context extension.
