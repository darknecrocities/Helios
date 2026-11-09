# HELIOS Continuous Pretraining & Improvement Telemetry Report

Generated: 2026-11-09 12:52 UTC · Budget: 30 min total · Category: `all` (177 External Datasets Synced) · Storage: `/Volumes/One Touch` (Verified External Drive)

| Tier | Mode | Device | Category | Steps (Prev → Total) | Loss Trajectory (Init → Prev → Curr / Gain) | Held-Out Loss | Held-Out PPL (Before → After) | Train tok/s | Infer tok/s | Checkpoint Health & SHA-256 | Context / CoT Budget |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | :--- | ---: |
| nano | full_parameter | mps | `all` | 3186 → **6298** (+3112) | 10.4375 → 1.2157 → 2.3318 (+77.7%) | 2.1257 | 19.39 → **8.38** (+56.8%) | 1736.2 | 10190.9 | HEALTHY (`3f036411ee72`) | 16,384 / 2,048 |
| alpha | full_parameter | mps | `all` | 628 → **1412** (+784) | 10.5483 → 3.4652 → 1.4382 (+86.4%) | 2.6534 | 76.45 → **14.20** (+81.4%) | 432.9 | 3274.0 | HEALTHY (`e0f927f1de1a`) | 32,768 / 4,096 |
| beta | full_parameter | mps | `all` | 178 → **585** (+407) | 10.6157 → 3.8918 → 2.9597 (+72.1%) | 3.4831 | 213.03 → **32.56** (+84.7%) | 107.4 | 1104.8 | HEALTHY (`0379395c0faa`) | 65,536 / 8,192 |
| core | block_slice | cpu | `all` | 1793 → **3803** (+2010) | 2016.0848 → 1.1983 → 3.0406 (+99.8%) | 3.7466 | 745.77 → **42.38** (+94.3%) | 783.6 | 6842.2 | HEALTHY (`db081d2c8024`) | 131,072 / 16,384 |
| pro | block_slice | cpu | `all` | 845 → **1691** (+846) | 3031.3721 → 8.4895 → 8.1789 (+99.7%) | 6.0116 | 131252.14 → **408.13** (+99.7%) | 327.9 | 3309.2 | HEALTHY (`a7d04f10e430`) | 131,072 / 24,576 |
| flagship | block_slice | cpu | `all` | 370 → **875** (+505) | 4053.9004 → 27.3593 → 3.8309 (+99.9%) | 11.5704 | 485165195.41 → **105913.58** (+100.0%) | 196.6 | 1975.7 | HEALTHY (`ec07d590bd6a`) | 262,144 / 32,768 |

## Last Checkpoint Persistence & How to Run in `helios chat`
- Every tier's latest verified checkpoint is atomically saved to `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/` (`model.safetensors`, `state.pt`, `config.json`, `manifest.json`, `SHA256SUMS`).
- Launching `helios chat -m <tier>` automatically restores the latest checkpoint from `/Volumes/One Touch` and verifies its SHA-256 integrity.
- Inside `helios chat`, run `/status` (or `/health`) to inspect corruption/health checks and before-vs-after improvement deltas, `/categories` to browse the 10 domain categories, `/train <tier> --category <cat> --minutes <N>` to continue training from the last checkpoint, and `/sync` to publish statistical updates to Hugging Face, Ollama, and Git.
