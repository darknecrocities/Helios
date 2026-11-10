# HELIOS — Trained Topics & Non-Repeating Curriculum Ledger

> **Zero-Duplication Guarantee**: Every `helios train` / `/train` run advances a persistent cursor in `/Volumes/One Touch/Helios/manifests/curriculum_ledger.json` and verifies SHA-256 record digests so consecutive training sessions never repeat the same dataset slice until a full epoch completes.

## 1. Cumulative Curriculum Status
- **Total External Datasets Synced**: `177` (`120,616` consolidated records)
- **Cumulative Unique Records Trained**: `21,132`
- **Latest Active Cursor (`all`)**: Offset `9,900` / `120,616` (Epoch `1`)
- **Latest Trained Topics Summary**: `Coding 64% · Math 18% · Agents 14% · Geography 1% · Tagalog 1% · History 1% · Science 1% · General Web 1% · Law 1%`
- **Cross-Session Duplication Rate**: `0.0%`

## 2. Cumulative Topics Trained Across All Sessions

| Topic Domain | Records / Samples Trained | Share | Duplication Rate |
| :--- | ---: | ---: | :--- |
| **Coding** | `8,718` | `60.1%` | `0% (SHA-256 Verified)` |
| **Math** | `2,628` | `18.1%` | `0% (SHA-256 Verified)` |
| **Agents** | `2,121` | `14.6%` | `0% (SHA-256 Verified)` |
| **Geography** | `230` | `1.6%` | `0% (SHA-256 Verified)` |
| **Science** | `224` | `1.5%` | `0% (SHA-256 Verified)` |
| **Tagalog** | `198` | `1.4%` | `0% (SHA-256 Verified)` |
| **General Web** | `197` | `1.4%` | `0% (SHA-256 Verified)` |
| **History** | `137` | `0.9%` | `0% (SHA-256 Verified)` |
| **Law** | `43` | `0.3%` | `0% (SHA-256 Verified)` |

## 3. Session-by-Session Non-Repeating Training Log

### Session #3 — `2026-11-10 03:33:06 UTC` (Category: `all`)
- **Corpus Slice**: `Rec #7,400→#9,900 (0% Dup · Ep 1)` (`7492` unique texts, `0` duplicates skipped)
- **Trained Topics Breakdown**: **Coding** (2504), **Math** (711), **Agents** (554), **Geography** (46), **Tagalog** (32), **History** (29), **Science** (17), **General Web** (11), **Law** (7)
- **Active External Datasets (177 mapped)**: `FineWeb-Edu (1.3T Tokens)`, `FineWeb (15T Tokens)`, `DCLM-Baseline (4T Tokens)`, `Cosmopedia v2`, `Wikipedia Multi-Domain Dump`, `Wikidata Knowledge Graph Triples`
- **Sample Prompts Trained in This Slice**:
  - *"Create an array containing 10 random numbers between 0 and 10."*
  - *"Write a code to edit a given column in an SQL table."*
  - *"How would you implement a simple object oriented Programming concept in Python?"*
  - *"Write a JavaScript function to perform a binary search.
let array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
let x = 5;"*

### Session #2 — `2026-11-09 15:56:23 UTC` (Category: `all`)
- **Corpus Slice**: `Rec #4,900→#7,400 (0% Dup · Ep 1)` (`7492` unique texts, `0` duplicates skipped)
- **Trained Topics Breakdown**: **Coding** (2504), **Math** (714), **Agents** (581), **Geography** (42), **Tagalog** (27), **History** (22), **Science** (13), **General Web** (11), **Law** (7)
- **Active External Datasets (177 mapped)**: `FineWeb-Edu (1.3T Tokens)`, `FineWeb (15T Tokens)`, `DCLM-Baseline (4T Tokens)`, `Cosmopedia v2`, `Wikipedia Multi-Domain Dump`, `Wikidata Knowledge Graph Triples`
- **Sample Prompts Trained in This Slice**:
  - *"Modify the following code to output Odd numbers from 1 to 10
for num in range(1, 10):
    print( num )"*
  - *"Given the following code, what should the output be when the function is called?
def add(x, y):
    z = x + y
"*
  - *"Using the given list, create a code that will generate a unique list of numbers
x = [4, 10, 16, 22, 28, 34]"*
  - *"What will the following code output?
i = 1

while i < 4:
    print(i)
    i = i + 1"*

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
