"""Reproducibility, Contamination Auditing, and Hardware Memory Profiling.

Verifies zero overlap between training corpora and held-out public evaluation
slices, computes cryptographic checkpoint digests, and calculates complete
runtime memory requirements (Q4 weights + KV cache + runtime buffers + RSS).

Author: AI Engineer Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import sys
from pathlib import Path
from typing import Any

import torch

from helios.model.config import HeliosConfig


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    if not path.is_file():
        return "not_present"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_word_ngrams(text: str, n: int = 13) -> set[str]:
    """Extract normalized word n-grams for benchmark contamination detection."""
    words = [w.strip(".,!?:;\"'()[]{}").lower() for w in text.split()]
    words = [w for w in words if w]
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def compute_hardware_memory_profile(
    config: HeliosConfig,
    context_window: int,
    runtime_overhead_mb: float = 64.0,
) -> dict[str, Any]:
    """Compute complete memory requirements: Q4 weights + KV cache + runtime buffers."""
    params = config.count_parameters()["total_parameters"]
    # 4-bit NormalFloat (NF4) weights: 0.5 bytes/param + 5% quantization block scales
    q4_weights_mb = (params * 0.525) / (1024**2)

    # FP16 KV-Cache size = 2 (K,V) * n_layers * n_kv_heads * head_dim * seq_len * 2 bytes
    kv_bytes_per_token = 2 * config.n_layers * config.n_kv_heads * config.head_dim * 2
    kv_cache_2k_mb = (kv_bytes_per_token * 2048) / (1024**2)
    kv_cache_max_ctx_mb = (kv_bytes_per_token * context_window) / (1024**2)

    total_2k_ram_mb = q4_weights_mb + kv_cache_2k_mb + runtime_overhead_mb
    total_max_ctx_ram_mb = q4_weights_mb + kv_cache_max_ctx_mb + runtime_overhead_mb

    # Measured OS peak RSS of current Python process (macOS reports bytes, Linux reports KB)
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    measured_rss_mb = raw_rss / (1024**2) if sys.platform == "darwin" else raw_rss / 1024.0

    return {
        "total_parameters": params,
        "context_window": context_window,
        "q4_weights_mb": round(q4_weights_mb, 2),
        "kv_cache_at_2k_mb": round(kv_cache_2k_mb, 2),
        "kv_cache_at_max_ctx_mb": round(kv_cache_max_ctx_mb, 2),
        "runtime_buffers_overhead_mb": round(runtime_overhead_mb, 2),
        "total_active_ram_2k_mb": round(total_2k_ram_mb, 2),
        "total_active_ram_max_ctx_mb": round(total_max_ctx_ram_mb, 2),
        "measured_process_peak_rss_mb": round(measured_rss_mb, 2),
    }


def run_contamination_audit(
    eval_slice_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Verify zero prompt or 13-gram overlap between training data and held-out test slices."""
    root = Path(__file__).resolve().parent.parent.parent
    slice_file = eval_slice_path or (
        root / "eval" / "datasets" / "public_test_slices" / "held_out_public_benchmarks.jsonl"
    )
    out_file = output_path or (
        root / "eval" / "datasets" / "provenance" / "contamination_audit.json"
    )

    training_sources = [
        root / "data" / "helios_expanded_reasoning_corpus.jsonl",
        Path("/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl"),
    ]

    train_prompt_hashes: set[str] = set()
    train_13grams: set[str] = set()
    scanned_records = 0

    for src in training_sources:
        if not src.is_file():
            continue
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                p = data.get("prompt", "").strip().lower().rstrip("?.!")
                if p:
                    scanned_records += 1
                    train_prompt_hashes.add(hashlib.sha256(p.encode("utf-8")).hexdigest())
                    if len(p.split()) >= 13:
                        train_13grams.update(extract_word_ngrams(p, n=13))

    eval_items = 0
    contaminated_ids: list[str] = []

    with open(slice_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            eval_items += 1
            p = item.get("prompt", "").strip().lower().rstrip("?.!")
            p_hash = hashlib.sha256(p.encode("utf-8")).hexdigest()
            item_ngrams = extract_word_ngrams(p, n=13) if len(p.split()) >= 13 else set()

            if p_hash in train_prompt_hashes or (item_ngrams & train_13grams):
                contaminated_ids.append(item["id"])

    report = {
        "status": "CLEAN" if not contaminated_ids else "CONTAMINATED",
        "eval_slice_file": str(slice_file.relative_to(root)),
        "eval_slice_sha256": compute_sha256(slice_file),
        "training_records_scanned": scanned_records,
        "eval_items_audited": eval_items,
        "contaminated_count": len(contaminated_ids),
        "contaminated_item_ids": contaminated_ids,
        "environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "platform": platform.platform(),
        },
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    audit = run_contamination_audit()
    print(json.dumps(audit, indent=2))
    if audit["contaminated_count"] > 0:
        sys.exit(1)
