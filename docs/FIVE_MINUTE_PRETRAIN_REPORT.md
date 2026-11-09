# HELIOS Continuous Pretraining & Improvement Telemetry Report

Generated: 2026-11-09 11:37 UTC · Budget: 20 min total · Category: `all` (177 External Datasets Synced) · Storage: `/Volumes/One Touch` (Verified External Drive)

| Tier | Mode | Device | Category | Steps (Prev → Total) | Loss Trajectory (Init → Prev → Curr / Gain) | Held-Out Loss | Held-Out PPL (Before → After) | Train tok/s | Infer tok/s | Checkpoint Health & SHA-256 | Context / CoT Budget |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | :--- | ---: |
| nano | full_parameter | mps | `all` | 516 → **3186** (+2670) | 10.4375 → 2.8134 → 1.2157 (+88.3%) | 2.9647 | **19.39** | 3416.6 | 9028.3 | HEALTHY (`b4a0af1fac86`) | 16,384 / 2,048 |
| alpha | full_parameter | mps | `all` | 156 → **628** (+472) | 10.5483 → 4.0479 → 3.4652 (+67.2%) | 4.3367 | **76.45** | 602.8 | 2736.1 | HEALTHY (`932d5cb4e15d`) | 32,768 / 4,096 |
| beta | full_parameter | mps | `all` | 46 → **178** (+132) | 10.6157 → 5.9297 → 3.8918 (+63.3%) | 5.3614 | **213.03** | 169.5 | 986.7 | HEALTHY (`5e9959304ea7`) | 65,536 / 8,192 |
| core | block_slice | cpu | `all` | 323 → **1793** (+1470) | 2016.0848 → 19.7864 → 1.1983 (+99.9%) | 6.6144 | **745.77** | 742.8 | 6710.8 | HEALTHY (`db30bdf7d084`) | 131,072 / 16,384 |
| pro | block_slice | cpu | `all` | 161 → **845** (+684) | 3031.3721 → 22.1783 → 8.4895 (+99.7%) | 11.7849 | **131252.14** | 347.0 | 3103.8 | HEALTHY (`07021f519eac`) | 131,072 / 24,576 |
| flagship | block_slice | cpu | `all` | 92 → **370** (+278) | 4053.9004 → 87.3585 → 27.3593 (+99.3%) | 21.7608 | **485165195.41** | 142.2 | 1776.4 | HEALTHY (`acac157dd1d6`) | 262,144 / 32,768 |

## Last Checkpoint Persistence & How to Run in `helios chat`
- Every tier's latest verified checkpoint is atomically saved to `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/` (`model.safetensors`, `state.pt`, `config.json`, `manifest.json`, `SHA256SUMS`).
- Launching `helios chat -m <tier>` automatically restores the latest checkpoint from `/Volumes/One Touch` and verifies its SHA-256 integrity.
- Inside `helios chat`, run `/status` (or `/health`) to inspect corruption/health checks and before-vs-after improvement deltas, `/categories` to browse the 10 domain categories, `/train <tier> --category <cat> --minutes <N>` to continue training from the last checkpoint, and `/sync` to publish statistical updates to Hugging Face, Ollama, and Git.
