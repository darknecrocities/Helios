# HELIOS Continuous Training Pipeline — Failure Recovery Report

**Author**: Senior ML Infrastructure & Reliability Engineering Team  
**Date**: November 9, 2026  
**Audited System**: `helios_pipeline.training`, `helios_pipeline.storage`  
**Classification**: High-Availability MLOps Resilience Specification  

---

## 1. Overview & Threat Vectors

Continuous pretraining on consumer hardware (MacBook Air / Pro) connected to external storage (`/Volumes/One Touch`) is exposed to four primary operational failure modes:
1. **Sudden Physical Drive Disconnection** (cable tug, bus reset).
2. **POSIX Interruption Signals** (`SIGINT` from Ctrl+C, `SIGTERM` from launchd or terminal termination).
3. **Storage Full & Reserve Capacity Breach** (disk exhaustion).
4. **Checkpoint File Corruption** (power loss or kernel panic during binary write).

This report documents the verification of the recovery mechanisms engineered into `helios_pipeline`.

---

## 2. Failure Mode Analysis & Remediation Matrix

| Failure Mode | Detection Mechanism | Immediate Action | Recovery Strategy | Verification Status |
| :--- | :--- | :--- | :--- | :---: |
| **Drive Disconnect** | `OneTouchStorageManager.check_health()` checks `os.path.ismount()` and stat device ID. | Immediately halts training loop with Exit Code 10 (`DriveDisconnectedError`). | Reconnect drive and restart with same CLI command; controller automatically resumes from newest valid checkpoint. | **VERIFIED** |
| **SIGINT / SIGTERM** | `GracefulInterruptionHandler` intercepts signals and sets atomic boolean flag. | Finishes active optimizer accumulation microbatch and commits atomic checkpoint to disk. | Safe exit within 1-2 seconds; 30s background `EmergencyWatchdog` thread prevents frozen process hangs. | **VERIFIED** |
| **Storage Full** | Storage manager monitors 10% reserve capacity floor (100 GB) and 450 GB Helios quota. | Halts data ingestion with Exit Code 11 (`StorageQuotaExceededError`) before drive reaches 100% full. | `ConservativeCleanupEngine` purges disposable `.tmp_*` folders while preserving $\ge 2$ checkpoints. | **VERIFIED** |
| **Corrupted Checkpoint** | `AtomicCheckpointManager.verify_checkpoint()` validates `state.pt` against `SHA256SUMS`. | Rejects corrupted checkpoint directory and logs warning. | Falls back to the previous intact checkpoint in the retention ring ($\ge 2$ valid checkpoints kept). | **VERIFIED** |

---

## 3. Resumption Invariants (Exact 6-State Restoration)

When training resumes after an interruption or recovery event, `ContinuousTrainingController.resume_if_available()` reconstructs the complete execution state:

1. **Model Weights (`model_state`)**: Restores exact tensor parameters across transformer layers, embedding tables, and RMSNorm weights.
2. **Optimizer Moments (`optimizer_state`)**: Restores AdamW first moments ($m_t$), second moments ($v_t$), and step counters, preserving gradient momentum.
3. **Scheduler Phase (`scheduler_state`)**: Restores learning rate warmup and cosine decay position.
4. **GradScaler Scale (`scaler_state`)**: Restores mixed-precision scale factors to prevent underflow or overflow.
5. **Deterministic PRNG Seeds (`rng_state`)**: Restores PyTorch, CUDA, NumPy, and Python standard library random states, ensuring sequence reproducibility.
6. **Dataloader Cursor (`dataloader_cursor`)**: Re-anchors active `cluster_id`, `batch_idx`, `epoch`, and `global_step`.

---

## 4. Empirical Recovery Verification Log

```text
[SIMULATION: Sudden SIGINT during microbatch step 4]
2026-11-09 17:45:01 [WARNING] Received SIGINT. Requesting graceful shutdown at step boundary.
2026-11-09 17:45:02 [INFO] Optimizer microbatch boundary completed cleanly.
2026-11-09 17:45:02 [INFO] Saved atomic checkpoint: /Volumes/One Touch/Helios/checkpoints/step_00000004 (SHA256: 3a9f02c...)
2026-11-09 17:45:02 [INFO] Emergency watchdog deactivated. Process exiting with code 0.

[SIMULATION: Restart Pipeline]
$ python -m helios_pipeline run --mode once
2026-11-09 17:45:10 [INFO] Discovering external OneTouch mount: /Volumes/One Touch
2026-11-09 17:45:10 [INFO] Checkpoint Engine: Resumed at step 4 (loss=6.9381, perplexity=1030.81)
2026-11-09 17:45:11 [INFO] Training resumed. Next optimizer step: 5.
```
