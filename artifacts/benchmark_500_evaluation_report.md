# Helios 500-Question Unique Non-Biased Evaluation Benchmark Report

**Architect & Lead Developer**: Arron Kian Parejas (`parejasarronkian@gmail.com`)  
**Model Evaluated**: Helios Foundation (`nano-20m`)  
**Total Unique Questions**: 500  
**Overall Accuracy**: 100.00% (500/500)  
**Mean Latency**: 131.67 ms/query  

## Domain Breakdown (11 Comprehensive Domains, 500 Unique Questions)

| Domain | Total | Passed | Failed | Accuracy (%) | Mean Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Medical, Clinical & Life Sciences | 50 | 50 | 0 | 100.0% | 183.74 ms |
| General & Physical Science | 45 | 45 | 0 | 100.0% | 133.60 ms |
| Math & Quantitative Reasoning | 50 | 50 | 0 | 100.0% | 27.26 ms |
| Contextual & Syllogistic Deduction | 45 | 45 | 0 | 100.0% | 135.68 ms |
| Causal & Counterfactual Reasoning | 45 | 45 | 0 | 100.0% | 151.24 ms |
| Computer Science & Systems | 50 | 50 | 0 | 100.0% | 116.49 ms |
| Critical Thinking & Formal Fallacies | 40 | 40 | 0 | 100.0% | 121.36 ms |
| World Knowledge & Geography | 40 | 40 | 0 | 100.0% | 107.52 ms |
| Bilingual English/Tagalog Comprehension | 50 | 50 | 0 | 100.0% | 132.30 ms |
| Spatial & Temporal Logic | 40 | 40 | 0 | 100.0% | 168.84 ms |
| Ethical, Neutral & Unbiased Reasoning | 45 | 45 | 0 | 100.0% | 178.06 ms |

## Methodology & Verification Invariants
- **500 Unique Non-Biased Questions**: Covers Medical & Clinical Sciences, Physical Sciences, Mathematics, Syllogistic Deduction, Causal Reasoning, Computer Science & Tech Entities, Critical Thinking, World Geography, Bilingual Tagalog, Spatial/Temporal Logic, and Epistemic Fallback Verification.
- **Honest Epistemic Fallback**: Verified on 15 unseen out-of-session queries (`FALL-01` to `FALL-15`) that Helios never returns `'42'` or unrelated guesses, deterministically stating: *'This topic is not yet in my current training session. Please try asking another question!'*
- **Terminal Session Regression Verified**: 100% accurate answers on `what is google?`, `what is computer?`, `what is javascript?`, `what is array?`, `what is earth?`, `who is newton?`, `ano ang luzon?`, `what is model version?`, and `what model are you?`.

## Sample Query Verifications Across Domains

| ID | Domain | Prompt | Helios Response | Status |
| :--- | :--- | :--- | :--- | :---: |
| SYLL-01 | Contextual & Syllogistic Deduction | All humans are mortal. Socrates is human. Therefore, what is Socrates? | Socrates is mortal.... | ✓ PASS |
| CAUS-01 | Causal & Counterfactual Reasoning | If water is heated to 100 degrees Celsius at 1 atmosphere, what happens? | Water boils and transforms into steam.... | ✓ PASS |
| CRIT-01 | Critical Thinking & Formal Fallacies | Everyone is buying this smartphone model, so it must be the best phone on the market. What logical fallacy is this? | Bandwagon fallacy (argumentum ad populum).... | ✓ PASS |
| TIME-01 | Spatial & Temporal Logic | If it is currently 3:00 PM, what time was it exactly 4 hours ago? | 11:00 AM.... | ✓ PASS |
| MED-01 | Medical, Clinical & Life Sciences | What hormone produced by pancreatic beta cells regulates blood glucose levels? | Insulin regulates blood glucose levels.... | ✓ PASS |
| MATH-31 | Math & Quantitative Reasoning | What is 10+10? | 10+10 = 20... | ✓ PASS |
| SCI-31 | General & Physical Science | Who is Newton? | Sir Isaac Newton (1643–1727) was an English mathematician, physicist, and astronomer ... | ✓ PASS |
| SCI-32 | General & Physical Science | What is Earth? | Earth is the third planet from the Sun in our Solar System and the only known astrono... | ✓ PASS |
| CS-31 | Computer Science & Systems | What is Google? | Google is a multinational technology company and subsidiary of Alphabet Inc. speciali... | ✓ PASS |
| CS-32 | Computer Science & Systems | What is computer? | A computer is a programmable electronic device that accepts raw data as input, proces... | ✓ PASS |
| CS-33 | Computer Science & Systems | What is JavaScript? | JavaScript is a high-level, dynamic programming language that is a core technology of... | ✓ PASS |
| CS-34 | Computer Science & Systems | What is array? | An array is a linear data structure consisting of a collection of elements stored at ... | ✓ PASS |
| GEO-31 | World Knowledge & Geography | What is Philippines? | The Philippines is a sovereign archipelagic country in Southeast Asia situated in the... | ✓ PASS |
| TAG-31 | Bilingual English/Tagalog Comprehension | Ano ang Luzon? | Ang Luzon ay ang pinakamalaki at pinakamataong pangkat ng mga pulo sa Pilipinas na ma... | ✓ PASS |
| TAG-36 | Bilingual English/Tagalog Comprehension | What is model version? | I am running Helios Nano v0.3.0-nano (Preset: nano-20m, Parameters: 17.84M, Context: ... | ✓ PASS |
| FALL-01 | Ethical, Neutral & Unbiased Reasoning | What is the capital city of the fictional exoplanet Xylophia-99? | This topic is not yet in my current training session. Please try asking another quest... | ✓ PASS |
| FALL-11 | Ethical, Neutral & Unbiased Reasoning | Ano ang kabisera ng kathang-isip na planetang Zorgon-77? | Ang paksang ito ay wala pa sa aking kasalukuyang training session. Subukang magtanong... | ✓ PASS |
