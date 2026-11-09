# HELIOS Time-Boxed Pretraining Report

Generated: 2026-11-09 10:26 UTC · Budget: 5 min total · Storage: /Volumes/One Touch (verified)

| Tier | Mode | Device | Steps | Train loss | Held-out loss | Held-out PPL | Train tok/s | Infer tok/s | Context / CoT budget |
| :--- | :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | :--- |
| nano | full_parameter | mps | 516 | 10.4375 → 2.8134 | 1.0629 | 2.89 | 3165.1 | 8669.2 | 16,384 / 2,048 |
| alpha | full_parameter | mps | 156 | 10.5483 → 4.0479 | 1.5891 | 4.9 | 963.0 | 2095.2 | 32,768 / 4,096 |
| beta | full_parameter | mps | 46 | 10.6157 → 5.9297 | 3.4222 | 30.64 | 276.6 | 742.8 | 65,536 / 8,192 |
| core | block_slice | cpu | 323 | 2016.0848 → 19.7864 | 5.8005 | 330.46 | 734.7 | 5791.8 | 131,072 / 16,384 |
| pro | block_slice | cpu | 161 | 3031.3721 → 22.1783 | 20.1487 | 485165195.41 | 364.3 | 2475.3 | 131,072 / 24,576 |
| flagship | block_slice | cpu | 92 | 4053.9004 → 87.3585 | 56.8533 | 485165195.41 | 208.3 | 1489.0 | 262,144 / 32,768 |

## Interpretation & limits
- `full_parameter` tiers (nano/alpha/beta) train the complete model.
- `block_slice` tiers (core/pro/flagship) train one transformer block at exact width with a
  2,048-token active vocab. Their loss is **not comparable** to full models and does not
  indicate full 1B–5.8B capability.
- A few minutes of training moves loss but does not produce usable language ability.
  Held-out perplexity in the thousands is expected at this stage.
