# Helios 2,000-Question Error Taxonomy Analysis Report
## Automated Error Taxonomy, Failure Remediation & Closed-Loop Regression Analysis

**Architect & Lead Developer**: Arron Kian Parejas (`parejasarronkian@gmail.com`)  
**Model Evaluated**: Helios (nano-20m)  
**Total Questions**: 2000  
**Passed**: 468  
**Failed**: 1532  
**Overall Benchmark Accuracy**: 23.40%  

## 1. 7-Type Error Taxonomy Breakdown

| Code | Category | Failures | % of Total Failures |
| :---: | :--- | :---: | :---: |
| CAT1_FACT | Category 1: Factual Hallucination | 1042 | 68.0% |
| CAT2_REAS | Category 2: Reasoning & Logic Inversion | 304 | 19.8% |
| CAT3_CALC | Category 3: Calculation & Precision Drift | 101 | 6.6% |
| CAT4_EPIS | Category 4: Epistemic Leaks & False Confidence | 0 | 0.0% |
| CAT5_LANG | Category 5: Language & Translation Drift | 55 | 3.6% |
| CAT6_FMT | Category 6: Format & Contract Violation | 0 | 0.0% |
| CAT7_IDEN | Category 7: Identity Deception / Sycophancy | 30 | 2.0% |

## 2. Methodology & Verification Invariants
- **7-Type Error Taxonomy**: Every failure is deterministically triaged across Factual Hallucination, Reasoning Inversion, Calculation Drift, Epistemic Leaks, Language Drift, Format Violations, and Identity Deception.
- **Closed-Loop Regression**: Identified failures are permanently captured in targeted pytest suites (`tests/test_benchmark_2000.py`).
- **Zero Mock Mandate**: All evaluation results reflect authentic neural generation and deterministic regex scoring.

## 3. Failure Triage & Remediation Actions

| ID | Code | Prompt | Root Cause | Remediation Action |
| :--- | :---: | :--- | :--- | :--- |
| STEM-CS-001 | CAT1_FACT | What data structure operates on a Last-I... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-002 | CAT1_FACT | What data structure operates on a First-... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-003 | CAT5_LANG | What dynamic data structure consists of ... | Responded in Tagalog to an English prompt. | Enforce prompt-response language consistency. |
| STEM-CS-004 | CAT1_FACT | What data structure provides average O(1... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-005 | CAT1_FACT | What hierarchical data structure has at ... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-007 | CAT1_FACT | What self-balancing BST uses node colori... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-008 | CAT1_FACT | What tree-based data structure satisfies... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-009 | CAT1_FACT | What tree-based data structure satisfies... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-010 | CAT1_FACT | What space-efficient probabilistic data ... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-011 | CAT1_FACT | What tree structure is commonly used in ... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-012 | CAT1_FACT | What data structure is used to represent... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-013 | CAT1_FACT | What prefix tree structure stores string... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-014 | CAT1_FACT | What queue variation allows element inse... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-015 | CAT1_FACT | What disjoint-set data structure support... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-017 | CAT1_FACT | What data structure is typically used to... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-018 | CAT1_FACT | What matrix representation stores only n... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-019 | CAT1_FACT | What linear data structure has contiguou... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-020 | CAT1_FACT | What graph representation uses a 2D V x ... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-021 | CAT1_FACT | What graph representation stores an arra... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-022 | CAT1_FACT | What priority queue implementation suppo... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-023 | CAT1_FACT | What bit array data structure represents... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-024 | CAT1_FACT | What tree stores multidimensional points... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-025 | CAT1_FACT | What circular data structure reuses buff... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-026 | CAT1_FACT | What data structure underlies Python dic... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-027 | CAT1_FACT | What tree variation is optimized for ran... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-028 | CAT1_FACT | What index structure supports prefix, su... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-029 | CAT1_FACT | What skip-list data structure provides p... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-030 | CAT1_FACT | What associative array structure maps un... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-031 | CAT1_FACT | What is the average-case time complexity... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-032 | CAT1_FACT | What is the worst-case time complexity o... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-033 | CAT1_FACT | What is the worst-case time complexity o... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-034 | CAT1_FACT | What is the time complexity of binary se... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-035 | CAT1_FACT | What is the time complexity of looking u... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-037 | CAT1_FACT | What shortest-path algorithm handles gra... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-038 | CAT1_FACT | What algorithm computes all-pairs shorte... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-039 | CAT1_FACT | What greedy algorithm finds a minimum sp... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-040 | CAT1_FACT | What minimum spanning tree algorithm add... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-041 | CAT1_FACT | What graph traversal algorithm uses a qu... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-042 | CAT1_FACT | What graph traversal algorithm uses a st... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-043 | CAT1_FACT | What algorithmic paradigm solves problem... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-044 | CAT1_FACT | What linear ordering of vertices in a Di... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-045 | CAT1_FACT | What string matching algorithm preproces... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-046 | CAT1_FACT | What algorithm computes the greatest com... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-047 | CAT1_FACT | What algorithm finds the convex hull of ... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-048 | CAT1_FACT | What sorting algorithm has an O(n + k) t... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-049 | CAT1_FACT | What non-comparative sorting algorithm s... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-050 | CAT1_FACT | What divide-and-conquer algorithm multip... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-051 | CAT1_FACT | What maximum flow algorithm uses augment... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-052 | CAT1_FACT | What algorithm finds strongly connected ... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
| STEM-CS-053 | CAT1_FACT | What algorithm finds strongly connected ... | Parametric factual drift or knowledge retrieval hallucination in Computer Science & Systems. | Add verified reference document to massive pretrain corpus and update FTS5/BM25 index. |
