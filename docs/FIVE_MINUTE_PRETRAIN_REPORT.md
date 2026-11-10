# HELIOS Continuous Pretraining & Improvement Telemetry Report

Generated: 2026-11-10 07:14 UTC · Budget: 0.5 min total · Category: `all` (177 External Datasets Synced) · Storage: `/Volumes/One Touch` (Verified External Drive)

| Tier | Mode | Device | Category | Steps (Prev → Total) | Loss Trajectory (Init → Prev → Curr / Gain) | Held-Out Loss | Held-Out PPL (Before → After) | Train tok/s | Infer tok/s | Checkpoint Health & SHA-256 | Context / CoT Budget |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | :--- | ---: |
| nano | full_parameter | mps | `all` | 14622 → **14667** (+45) | 10.4375 → 2.7989 → 0.0395 (+99.6%) | 1.0586 | **2.88** | 1700.3 | 10599.3 | HEALTHY (`c6d4949696a1`) | 16,384 / 2,048 |
| alpha | full_parameter | mps | `all` | 3900 → **3918** (+18) | 10.5483 → 0.0447 → 1.6051 (+84.8%) | 1.3183 | **3.74** | 667.0 | 3202.5 | HEALTHY (`e1f4f8f95c25`) | 32,768 / 4,096 |
| beta | full_parameter | mps | `all` | 1717 → **1726** (+9) | 10.6157 → 0.0981 → 0.1670 (+98.4%) | 1.798 | **6.04** | 146.2 | 1138.2 | HEALTHY (`146d55f8b30e`) | 65,536 / 8,192 |
| core | block_slice | cpu | `all` | 9210 → **9247** (+37) | 2016.0848 → 0.0997 → 0.2543 (+100.0%) | 1.8326 | **6.25** | 915.7 | 7621.5 | HEALTHY (`0a80a9923efb`) | 131,072 / 16,384 |
| pro | block_slice | cpu | `all` | 4142 → **4160** (+18) | 3031.3721 → 6.3637 → 5.3984 (+99.8%) | 3.0858 | **21.89** | 442.7 | 3523.2 | HEALTHY (`5c40a9f8d3ff`) | 131,072 / 24,576 |
| flagship | block_slice | cpu | `all` | 2390 → **2401** (+11) | 4053.9004 → 1.7754 → 3.7338 (+99.9%) | 6.6328 | **759.62** | 253.4 | 1973.6 | HEALTHY (`9e2faa21437f`) | 262,144 / 32,768 |

## Last Checkpoint Persistence & How to Run in `helios chat`
- Every tier's latest verified checkpoint is atomically saved to `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/` (`model.safetensors`, `state.pt`, `config.json`, `manifest.json`, `SHA256SUMS`).
- Launching `helios chat -m <tier>` automatically restores the latest checkpoint from `/Volumes/One Touch` and verifies its SHA-256 integrity.
- Inside `helios chat`, run `/status` (or `/health`) to inspect corruption/health checks and before-vs-after improvement deltas, `/categories` to browse the 10 domain categories, `/train <tier> --category <cat> --minutes <N>` to continue training from the last checkpoint, and `/sync` to publish statistical updates to Hugging Face, Ollama, and Git.
