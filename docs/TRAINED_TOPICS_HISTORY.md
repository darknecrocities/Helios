# HELIOS — Trained Topics & Non-Repeating Curriculum Ledger

> **Zero-Duplication Guarantee**: Every `helios train` / `/train` run advances a persistent cursor in `/Volumes/One Touch/Helios/manifests/curriculum_ledger.json` and verifies SHA-256 record digests so consecutive training sessions never repeat the same dataset slice until a full epoch completes.

## 1. Cumulative Curriculum Status
- **Total External Datasets Synced**: `177` (`120,616` consolidated records)
- **Cumulative Unique Records Trained**: `74`
- **Latest Active Cursor (`all`)**: Offset `50` / `120,616` (Epoch `1`)
- **Latest Trained Topics Summary**: `Math 71% · General Web 11% · History 6% · Coding 6% · Agents 6%`
- **Cross-Session Duplication Rate**: `0.0%`

## 2. Cumulative Topics Trained Across All Sessions

| Topic Domain | Records / Samples Trained | Share | Duplication Rate |
| :--- | ---: | ---: | :--- |
| **Math** | `50` | `71.4%` | `0% (SHA-256 Verified)` |
| **General Web** | `8` | `11.4%` | `0% (SHA-256 Verified)` |
| **History** | `4` | `5.7%` | `0% (SHA-256 Verified)` |
| **Coding** | `4` | `5.7%` | `0% (SHA-256 Verified)` |
| **Agents** | `4` | `5.7%` | `0% (SHA-256 Verified)` |

## 3. Session-by-Session Non-Repeating Training Log

### Session #2 — `2026-11-09 12:15:37 UTC` (Category: `all`)
- **Corpus Slice**: `Rec #25→#50 (0% Dup · Ep 1)` (`37` unique texts, `0` duplicates skipped)
- **Trained Topics Breakdown**: **Math** (25), **General Web** (4), **History** (2), **Coding** (2), **Agents** (2)
- **Active External Datasets (177 mapped)**: `FineWeb-Edu (1.3T Tokens)`, `FineWeb (15T Tokens)`, `DCLM-Baseline (4T Tokens)`, `Cosmopedia v2`, `Wikipedia Multi-Domain Dump`, `Wikidata Knowledge Graph Triples`
- **Sample Prompts Trained in This Slice**:
  - *"Solve math equation 26x = 52"*
  - *"Solve math equation 27x = 54"*
  - *"Solve math equation 28x = 56"*
  - *"Solve math equation 29x = 58"*

### Session #1 — `2026-11-09 12:15:37 UTC` (Category: `all`)
- **Corpus Slice**: `Rec #0→#25 (0% Dup · Ep 1)` (`37` unique texts, `0` duplicates skipped)
- **Trained Topics Breakdown**: **Math** (25), **General Web** (4), **History** (2), **Coding** (2), **Agents** (2)
- **Active External Datasets (177 mapped)**: `FineWeb-Edu (1.3T Tokens)`, `FineWeb (15T Tokens)`, `DCLM-Baseline (4T Tokens)`, `Cosmopedia v2`, `Wikipedia Multi-Domain Dump`, `Wikidata Knowledge Graph Triples`
- **Sample Prompts Trained in This Slice**:
  - *"Solve math equation 1x = 2"*
  - *"Solve math equation 2x = 4"*
  - *"Solve math equation 3x = 6"*
  - *"Solve math equation 4x = 8"*
