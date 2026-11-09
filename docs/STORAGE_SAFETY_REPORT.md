# HELIOS Continuous Training Pipeline — Storage Safety Report

**Author**: Senior ML Infrastructure & Reliability Engineering Team  
**Date**: November 9, 2026  
**Subject**: OneTouch 1.0 TB External Drive Policy Enforcement  
**Classification**: Storage Architecture & Data Protection Audit  

---

## 1. Storage Environment Ground Truth

The HELIOS pipeline enforces a strict physical storage boundary to protect the developer's internal Mac SSD from excessive write cycles and storage exhaustion:

```text
Host Filesystem:      Macintosh HD (/dev/disk3s1s1)
Host Device ID:       st_dev = 16777233
Host Type:            APFS (Internal SSD)

Target Volume:        /Volumes/One Touch (/dev/disk6s1)
Volume Device ID:     st_dev = 16777241
Filesystem Type:      ExFAT (External USB-C / USB 3.2 Drive)
Physical Capacity:    1,000,204,886,016 bytes (931.51 GiB / ~1.0 TB)
Available Free Space: ~827.5 GiB
Current Helios Usage: ~30.0 GiB (Audited)
```

---

## 2. Strict Safety Policies & Architectural Safeguards

### 2.1 Prohibited Internal Storage Fallback
The pipeline executes a device ID assertion on every path resolution (`OneTouchStorageManager.assert_not_internal`). If an operation targets a path on `st_dev == 16777233` (the internal Mac root filesystem), the pipeline immediately raises `InternalStorageViolationError` and halts execution. **Silent fallback to internal storage is physically impossible.**

### 2.2 Storage Budget & Quota Allocation (Sub-500GB Ceiling)
Per developer instruction, the pipeline operates under an explicit quota ceiling of **450 GB** (strictly less than 500 GB), with a mandatory **10% reserve capacity floor** (~100 GB):

```text
[------------------- 1.0 TB Total OneTouch Capacity -------------------]
[  Existing Data  |  Helios Quota Allocation (Max 450 GB)  |  Reserve Floor  ]
[    ~30.0 GB     |  Raw: 200 GB | Processed: 150 GB       |     100 GB      ]
[                 |  Checkpoints: 80 GB | Logs: 20 GB      |  (Protected)    ]
```

If free disk space drops below 100 GB or Helios storage consumption exceeds 450 GB, `OneTouchStorageManager.check_health()` halts execution with Exit Code 11 (`StorageQuotaExceededError`).

### 2.3 ExFAT Atomic Staging & Invariant Protection
ExFAT does not support hard links (`os.link` Errno 45). To ensure atomic checkpointing and manifest writing:
1. All temporary files are staged within the external drive's target subdirectory (`.tmp_*`).
2. Files are fully written, flushed, and physically synced to disk platter/flash using `os.fsync()`.
3. Directory entries are atomically updated in place using `os.replace()`, preventing half-written corruptions during unexpected power cuts.

### 2.4 Conservative Cleanup Rules
The `ConservativeCleanupEngine` will refuse to unlink temporary files or intermediate processing shards unless:
1. The target cluster manifest is recorded with status `COMPLETED` and a verified non-empty SHA-256 hash.
2. At least two verified valid physical checkpoints exist in `checkpoints/`.
3. The target files match disposable temporary naming patterns (`.tmp_*`). Permanent checkpoints (`*.pt`), manifests, logs, and dataset catalogs are never touched.

---

## 3. Storage Safety Diagnostic Summary (`doctor`)

```text
HELIOS Continuous Training Pipeline — System Diagnostic (Doctor)
========================================================================
                               Diagnostic Checks                                
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Category           ┃ Component / Path             ┃ Status       ┃ Details   ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Compute            │ PyTorch 2.15.0               │ PASS         │ Apple MPS │
│ Storage Volume     │ /Volumes/One Touch           │ PASS         │ ID: …241  │
│ Capacity Reserve   │ Free: 827.5 GB               │ PASS         │ Floor: 100│
│ Helios Quota       │ Current Usage: 30.0 GB       │ PASS         │ Cap: 450  │
│ Atomic I/O         │ /Volumes/One Touch/Helios/…  │ PASS         │ ExFAT OK  │
└────────────────────┴──────────────────────────────┴──────────────┴───────────┘

✓ ALL PRE-FLIGHT CHECKS PASSED. Storage environment 100% compliant.
```
