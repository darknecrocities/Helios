# HELIOS — Trained Topics & Non-Repeating Curriculum Ledger

> **Zero-Duplication Guarantee**: Every `helios train` / `/train` run advances a persistent cursor in `/Volumes/One Touch/Helios/manifests/curriculum_ledger.json` and verifies SHA-256 record digests so consecutive training sessions never repeat the same dataset slice until a full epoch completes.

## 1. Cumulative Curriculum Status
- **Total External Datasets Synced**: `177` (`120,616` consolidated records)
- **Cumulative Unique Records Trained**: `6,148`
- **Latest Active Cursor (`all`)**: Offset `4,900` / `120,616` (Epoch `1`)
- **Latest Trained Topics Summary**: `Coding 62% · Math 19% · Agents 16% · Geography 1% · Tagalog 1%`
- **Cross-Session Duplication Rate**: `0.0%`

## 2. Cumulative Topics Trained Across All Sessions

| Topic Domain | Records / Samples Trained | Share | Duplication Rate |
| :--- | ---: | ---: | :--- |
| **Coding** | `3,710` | `55.7%` | `0% (SHA-256 Verified)` |
| **Math** | `1,203` | `18.1%` | `0% (SHA-256 Verified)` |
| **Agents** | `986` | `14.8%` | `0% (SHA-256 Verified)` |
| **Science** | `194` | `2.9%` | `0% (SHA-256 Verified)` |
| **General Web** | `175` | `2.6%` | `0% (SHA-256 Verified)` |
| **Geography** | `142` | `2.1%` | `0% (SHA-256 Verified)` |
| **Tagalog** | `139` | `2.1%` | `0% (SHA-256 Verified)` |
| **History** | `86` | `1.3%` | `0% (SHA-256 Verified)` |
| **Law** | `29` | `0.4%` | `0% (SHA-256 Verified)` |

## 3. Session-by-Session Non-Repeating Training Log

### Session #1 — `2026-11-09 12:16:45 UTC` (Category: `all`)
- **Corpus Slice**: `Rec #2,400→#4,900 (0% Dup · Ep 1)` (`3748` unique texts, `0` duplicates skipped)
- **Trained Topics Breakdown**: **Coding** (2502), **Math** (755), **Agents** (653), **Geography** (58), **Tagalog** (27), **History** (19), **Science** (14), **General Web** (4)
- **Active External Datasets (177 mapped)**: `FineWeb-Edu (1.3T Tokens)`, `FineWeb (15T Tokens)`, `DCLM-Baseline (4T Tokens)`, `Cosmopedia v2`, `Wikipedia Multi-Domain Dump`, `Wikidata Knowledge Graph Triples`
- **Sample Prompts Trained in This Slice**:
  - *"Create an array of column names in a MySQL query.
Column Names: item_name, item_price"*
  - *"Write a function in PHP to convert a string to an array.
String: "A,B,C""*
  - *"Describe how a bubble sort algorithm works."*
  - *"Create a while loop in JavaScript to print out all the numbers between 20 and 30."*
