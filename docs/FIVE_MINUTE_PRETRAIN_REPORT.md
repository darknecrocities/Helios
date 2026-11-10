# HELIOS Continuous Pretraining & Improvement Telemetry Report

Generated: 2026-11-10 03:39 UTC · Budget: 1 min total · Category: `all` (177 External Datasets Synced) · Storage: `/Volumes/One Touch` (Verified External Drive)

| Tier | Mode | Device | Category | Steps (Prev → Total) | Loss Trajectory (Init → Prev → Curr / Gain) | Held-Out Loss | Held-Out PPL (Before → After) | Train tok/s | Infer tok/s | Checkpoint Health & SHA-256 | Context / CoT Budget |
| :--- | :--- | :--- | :--- | ---: | :--- | ---: | :--- | ---: | ---: | :--- | ---: |
| nano | full_parameter | mps | `all` | 12685 → **12785** (+100) | 10.4375 → 1.9111 → 0.0460 (+99.6%) | 0.8852 | 3.28 → **2.42** (+26.2%) | 1653.2 | 8765.3 | HEALTHY (`91939b57ef50`) | 16,384 / 2,048 |
| alpha | full_parameter | mps | `all` | 3366 → **3403** (+37) | 10.5483 → 0.0484 → 0.0597 (+99.4%) | 1.1093 | 4.63 → **3.03** (+34.6%) | 620.1 | 2975.4 | HEALTHY (`b50e9e1fd0c0`) | 32,768 / 4,096 |
| beta | full_parameter | mps | `all` | 1459 → **1479** (+20) | 10.6157 → 0.1355 → 5.9101 (+44.3%) | 1.5816 | 6.46 → **4.86** (+24.8%) | 157.5 | 1050.2 | HEALTHY (`e45edf2414b5`) | 65,536 / 8,192 |
| core | block_slice | cpu | `all` | 8036 → **8110** (+74) | 2016.0848 → 0.3438 → 2.6278 (+99.9%) | 1.9273 | 7.53 → **6.87** (+8.8%) | 907.0 | 7440.1 | HEALTHY (`2880196c3c95`) | 131,072 / 16,384 |
| pro | block_slice | cpu | `all` | 3595 → **3633** (+38) | 3031.3721 → 7.7093 → 4.5457 (+99.8%) | 3.3108 | 23.40 → **27.41** (-17.1%) | 455.5 | 3610.1 | HEALTHY (`05f65c008b51`) | 131,072 / 24,576 |
| flagship | block_slice | cpu | `all` | 2097 → **2120** (+23) | 4053.9004 → 5.0951 → 0.1892 (+100.0%) | 5.7503 | 399.35 → **314.28** (+21.3%) | 277.1 | 1980.1 | HEALTHY (`ef85adf032fd`) | 262,144 / 32,768 |

## Last Checkpoint Persistence & How to Run in `helios chat`
- Every tier's latest verified checkpoint is atomically saved to `/Volumes/One Touch/Helios/checkpoints/<tier>_continuous/latest/` (`model.safetensors`, `state.pt`, `config.json`, `manifest.json`, `SHA256SUMS`).
- Launching `helios chat -m <tier>` automatically restores the latest checkpoint from `/Volumes/One Touch` and verifies its SHA-256 integrity.
- Inside `helios chat`, run `/status` (or `/health`) to inspect corruption/health checks and before-vs-after improvement deltas, `/categories` to browse the 10 domain categories, `/train <tier> --category <cat> --minutes <N>` to continue training from the last checkpoint, and `/sync` to publish statistical updates to Hugging Face, Ollama, and Git.
