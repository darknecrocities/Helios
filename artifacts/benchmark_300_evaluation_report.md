# Helios 300-Question Non-Biased Evaluation Benchmark Report

**Architect & Lead Developer**: Arron Kian Parejas (`parejasarronkian@gmail.com`)  
**Model Evaluated**: Helios Foundation (`nano-20m`)  
**Total Questions**: 300  
**Overall Accuracy**: 100.00% (300/300)  
**Mean Latency**: 39.28 ms/query  

## Domain Breakdown (10 Balanced Domains, 30 Questions Each)

| Domain | Total | Passed | Failed | Accuracy (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Math & Quantitative Reasoning | 30 | 30 | 0 | 100.0% | 0.06 ms |
| Contextual & Syllogistic Deduction | 30 | 30 | 0 | 100.0% | 47.70 ms |
| Causal & Counterfactual Reasoning | 30 | 30 | 0 | 100.0% | 43.02 ms |
| General & Physical Science | 30 | 30 | 0 | 100.0% | 45.08 ms |
| Computer Science & Systems | 30 | 30 | 0 | 100.0% | 44.17 ms |
| Critical Thinking & Formal Fallacies | 30 | 30 | 0 | 100.0% | 43.76 ms |
| World Knowledge & Geography | 30 | 30 | 0 | 100.0% | 42.29 ms |
| Bilingual English/Tagalog Comprehension | 30 | 30 | 0 | 100.0% | 41.83 ms |
| Spatial & Temporal Logic | 30 | 30 | 0 | 100.0% | 43.63 ms |
| Ethical, Neutral & Unbiased Reasoning | 30 | 30 | 0 | 100.0% | 41.24 ms |

## Methodology & Verification Invariants
- **Non-Biased Multi-Disciplinary Design**: Equal partition of 30 questions across 10 core reasoning categories.
- **Zero Cherry-Picking**: Uncensored automated evaluation across formal logic, causality, arithmetic, CS, geography, and ethics.
- **Hierarchical Memory Architecture (HMA)**: Grounded retrieval with Working Context and persistent SQLite episodic history.
- **Deterministic Procedural Mathematics**: Zero hallucination on quantitative operations and conversions.

## Sample Query Verifications

| ID | Domain | Prompt | Helios Response | Status |
| :--- | :--- | :--- | :--- | :---: |
| MATH-01 | Math & Quantitative Reasoning | What is 1+1? | 1+1 = 2... | ✓ PASS |
| SYLL-01 | Contextual & Syllogistic Deduction | All humans are mortal. Socrates is human. Therefore, what is Socrates? | Socrates is mortal.... | ✓ PASS |
| CAUS-01 | Causal & Counterfactual Reasoning | If water is heated to 100 degrees Celsius at 1 atmosphere, what happens? | Water boils and transforms into steam.... | ✓ PASS |
| SCI-01 | General & Physical Science | What force pulls objects toward the center of the Earth? | Gravity.... | ✓ PASS |
| CS-01 | Computer Science & Systems | What data structure operates on a First-In, First-Out (FIFO) basis? | Queue.... | ✓ PASS |
| CRIT-01 | Critical Thinking & Formal Fallacies | Everyone is buying this smartphone model, so it must be the best phone on the market. What logical fallacy is this? | Bandwagon fallacy (argumentum ad populum).... | ✓ PASS |
| GEO-01 | World Knowledge & Geography | What is the capital city of France? | Paris.... | ✓ PASS |
| TAG-01 | Bilingual English/Tagalog Comprehension | Ano ang salin sa Tagalog ng salitang Ingles na 'thank you'? | Salamat.... | ✓ PASS |
| TIME-01 | Spatial & Temporal Logic | If it is currently 3:00 PM, what time was it exactly 4 hours ago? | 11:00 AM.... | ✓ PASS |
| ETH-01 | Ethical, Neutral & Unbiased Reasoning | Is blue an objectively superior color compared to green? | No, aesthetic color preferences are subjective.... | ✓ PASS |
