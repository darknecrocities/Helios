# HELIOS Continuous Pretraining, Category Selection, Last-Checkpoint Resumption & `helios chat` Guide

**Branch**: `pretrained` · **Persistent Volume**: `/Volumes/One Touch/Helios` · **Synced Datasets**: `177 External Datasets (120,616 Records / 90.38M Tokens)`

---

## 1. Overview & Strict Storage Guarantee

All continuous pretraining runs are hard-locked to the external **`/Volumes/One Touch`** drive (`st_dev != 16777233`):
- If `/Volumes/One Touch` is unmounted or unavailable, training halts immediately with **Exit Code 10** and never falls back to internal Mac storage.
- Every training run **automatically resumes from the last saved checkpoint** on `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/` rather than resetting weights.

---

## 2. Selectable Pretraining Categories (`--category` / `-c`)

| Category Key | Aliases | Domain Coverage | External Datasets | OneTouch Records |
| :--- | :--- | :--- | ---: | ---: |
| `all` | `omni`, `default` | Full 16-domain pretraining mix across all 177 datasets | 177 | 120,616 |
| `fineweb` | `web`, `encyclopedia` | `HuggingFaceFW/fineweb`, `fineweb-edu`, `dCLM-baseline`, `cosmopedia-v2`, `wikipedia`, RFCs | 38 | 11,248 |
| `math` | `mathematics` | `GSM8K`, `MetaMathQA`, `MATH`, `ProofNet`, `NuminaMath` | 14 | 17,473 |
| `history` | `hist` | World History Timelines, Ancient Civilizations, Archaeology, MMLU History | 22 | 4,465 |
| `coding` | `code`, `swe`, `python` | `The Stack v2`, `StarCoderData`, `Magicoder`, `CodeAlpaca`, Python SWE | 26 | 53,628 |
| `science` | `bio`, `medical`, `nature` | `SciQ`, `PubMedQA`, `MedMCQA`, `PeS2o`, Biology, Genetics, Ecology | 33 | 23,798 |
| `geography` | `geo`, `places` | World Geography, Capitals, Philippine 17 Administrative Regions | 10 | 1,248 |
| `law` | `legal`, `philosophy` | Constitutional Law, Jurisprudence, Governance, Formal Ethics | 10 | 1,248 |
| `agents` | `mcp`, `agent`, `tools` | Agentic Skills, Model Context Protocol (`MCP`), `BFCL` Function Calling | 16 | 10,004 |
| `tagalog` | `filipino`, `ph` | Bilingual Filipino / Tagalog & Cross-Lingual Comprehension | 8 | 1,248 |

---

## 3. How to Pretrain & Auto-Sync from Terminal CLI

```bash
# Train all 6 tiers for 20 minutes on the full 177-dataset omni-domain corpus
# Automatically benchmarks, updates docs/, stamps Ollama Modelfiles, pushes 7 Ollama tags,
# uploads statistical telemetry to Hugging Face (Darknecrocities/Helios), and pushes to Git
uv run helios train all --minutes 20 --category all

# Train on FineWeb & Encyclopedic Web data
uv run helios train all --minutes 20 --category fineweb

# Train specific tiers on Math or History
uv run helios train nano,alpha --minutes 10 --category math
uv run helios train beta --minutes 10 --category history
```

---

## 4. How the Last Checkpoint Is Saved & Loaded in `helios chat`

After every pretraining run on tier `<tier>`, HELIOS atomically writes:
1. `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/step_<NNNNNNNN>/`
2. `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/`
3. `/Volumes/One Touch/Helios/checkpoints/helios_<tier>/` (for full-parameter tiers `nano`, `alpha`, `beta`)

Each checkpoint bundle contains:
- `model.safetensors` & `state.pt` (verified `0 NaN / 0 Inf`)
- `config.json` (exact `HeliosConfig` architecture specification)
- `manifest.json` (step count, category, initial/previous/final loss, `% improvement`, held-out perplexity before/after, throughput, SHA-256 digest, `HEALTHY` status)
- `SHA256SUMS` (cryptographic digest of `model.safetensors`)

### Running in `helios chat`
```bash
# Automatically loads /Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/
uv run helios chat
uv run helios chat -m alpha

# Or load a specific checkpoint directory explicitly:
uv run helios chat -m nano --model-path "/Volumes/One Touch/Helios/checkpoints/nano_continuous/latest"
```

---

## 5. Interactive `helios chat` Telemetry, Health Audit & Auto-Sync Commands

Inside `helios chat`, use these slash commands at any time:
- `/status` (or `/health`): Displays the 6-tier **Checkpoint Health, Corruption Audit & Improvement Telemetry** table (`● HEALTHY` / `✗ CORRUPTED` / `○ UNINIT`, SHA-256 check, tensor `NaN`/`Inf` count, `Resumed → Total Steps`, `Loss Init → Prev → Curr (% Gain)`, and `Held-Out PPL Before → After (% Reduction)`).
- `/categories`: Displays all 10 pretraining dataset categories with dataset counts, record counts, and ready-to-run `/train` commands.
- `/train [tier] [--category <cat>] [--minutes <N>]`: Runs continuous pretraining from the last checkpoint, displays the improvement table, auto-syncs to Hugging Face, Ollama, and Git with statistical descriptions, and hot-reloads the active chat model live.
- `/sync` (or `/publish`, `/update`): Immediately stamps Ollama Modelfiles with live statistical telemetry, pushes all 7 Ollama tags (`parejasarronkian/Helios:<tier>`), uploads telemetry reports & manifests to Hugging Face (`Darknecrocities/Helios`), and commits/pushes `docs/` to Git.
- `/stats`: Displays session generation throughput alongside the active model's checkpoint health, SHA-256 digest, trained category, and loss/perplexity improvement delta.
- `/doctor`: Runs full system diagnostics on `/Volumes/One Touch`, NPU/MPS compute backends, and checkpoint health.
