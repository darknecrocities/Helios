# HELIOS Continuous Pretraining & Improvement Telemetry Report

Generated: 2026-11-09 17:02 UTC · Budget: 60 min total · Category: `all` (177 External Datasets Synced) · Storage: `/Volumes/One Touch` (Verified External Drive)

| Tier | Mode | Device | Category | Steps (Prev → Total) | Loss Trajectory (Init → Prev → Curr / Gain) | Held-Out Loss | Held-Out PPL (Before → After) | Train tok/s | Infer tok/s | Checkpoint Health & SHA-256 | Context / CoT Budget |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | :--- | ---: |
| nano | full_parameter | mps | `all` | 6298 → **12685** (+6387) | 10.4375 → 2.3318 → 1.9111 (+81.7%) | 1.1888 | 8.38 → **3.28** (+60.9%) | 1790.1 | 10160.0 | HEALTHY (`c578366d1cee`) | 16,384 / 2,048 |
| alpha | full_parameter | mps | `all` | 1412 → **3366** (+1954) | 10.5483 → 1.4382 → 0.0484 (+99.5%) | 1.5335 | 14.20 → **4.63** (+67.4%) | 544.0 | 3193.5 | HEALTHY (`529922b3bbd0`) | 32,768 / 4,096 |
| beta | full_parameter | mps | `all` | 585 → **1459** (+874) | 10.6157 → 2.9597 → 0.1355 (+98.7%) | 1.8652 | 32.56 → **6.46** (+80.2%) | 120.0 | 1114.2 | HEALTHY (`3cc7fcce7d6e`) | 65,536 / 8,192 |
| core | block_slice | cpu | `all` | 3803 → **8036** (+4233) | 2016.0848 → 3.0406 → 0.3438 (+100.0%) | 2.019 | 42.38 → **7.53** (+82.2%) | 864.5 | 8154.1 | HEALTHY (`24583ebadc3c`) | 131,072 / 16,384 |
| pro | block_slice | cpu | `all` | 1691 → **3595** (+1904) | 3031.3721 → 8.1789 → 7.7093 (+99.8%) | 3.1527 | 408.13 → **23.40** (+94.3%) | 389.0 | 3613.3 | HEALTHY (`9982f18782e0`) | 131,072 / 24,576 |
| flagship | block_slice | cpu | `all` | 875 → **2097** (+1222) | 4053.9004 → 3.8309 → 5.0951 (+99.9%) | 5.9898 | 105913.58 → **399.35** (+99.6%) | 249.8 | 2142.6 | HEALTHY (`5776d786964f`) | 262,144 / 32,768 |

## Last Checkpoint Persistence & How to Run in `helios chat`
- Every tier's latest verified checkpoint is atomically saved to `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/` (`model.safetensors`, `state.pt`, `config.json`, `manifest.json`, `SHA256SUMS`).
- Launching `helios chat -m <tier>` automatically restores the latest checkpoint from `/Volumes/One Touch` and verifies its SHA-256 integrity.
- Inside `helios chat`, run `/status` (or `/health`) to inspect corruption/health checks and before-vs-after improvement deltas, `/categories` to browse the 10 domain categories, `/train <tier> --category <cat> --minutes <N>` to continue training from the last checkpoint, and `/sync` to publish statistical updates to Hugging Face, Ollama, and Git.
