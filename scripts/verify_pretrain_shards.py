#!/usr/bin/env python3
"""Verify tokenized pretraining shards against their dataset manifest.

Checks
------
- Shard byte size equals ``tokens * 2`` for uint16.
- SHA-256 matches the manifest.
- Every token ID is inside the tokenizer vocabulary.
- Provenance line count equals the manifest document count.
- Provenance hashes are unique within and across train/val splits.
- No shard file is duplicated.
- Totals match the manifest stats.

This verifier cannot re-read raw text because shards contain tokens only. Text
quality and contamination filtering are enforced at build time and recorded in
the manifest; this script verifies the stored artifact is intact and that
deduplication did not leave hash collisions across splits.

Usage
-----
    python scripts/verify_pretrain_shards.py \
        --dataset-dir "/Volumes/One Touch/Helios/datasets/helios_fineweb_v1" \
        --json-out reports/fineweb_shard_audit.json

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(dataset_dir: Path, vocab_size: int | None) -> dict[str, Any]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_dir = dataset_dir / "shards"

    problems: list[str] = []
    shard_audits: list[dict[str, Any]] = []
    all_hashes: dict[str, str] = {}
    split_hashes: dict[str, set[str]] = {"train": set(), "val": set()}
    seen_shard_sha: set[str] = set()
    total_tokens = 0
    total_docs = 0

    for shard in manifest.get("shards", []):
        path = shard_dir / shard["file"]
        provenance = shard_dir / shard["provenance"]
        if not path.is_file():
            problems.append(f"{shard['file']}: missing shard file")
            continue
        if not provenance.is_file():
            problems.append(f"{shard['file']}: missing provenance file")
            continue

        size = path.stat().st_size
        expected_size = int(shard["tokens"]) * 2
        digest = sha256_file(path)
        values = np.fromfile(path, dtype="<u2")
        minimum = int(values.min()) if values.size else 0
        maximum = int(values.max()) if values.size else 0
        token_count = int(values.size)

        if size != expected_size:
            problems.append(f"{shard['file']}: size {size} != {expected_size}")
        if token_count != int(shard["tokens"]):
            problems.append(f"{shard['file']}: tokens {token_count} != manifest {shard['tokens']}")
        if digest != shard["sha256"]:
            problems.append(f"{shard['file']}: SHA-256 mismatch")
        if digest in seen_shard_sha:
            problems.append(f"{shard['file']}: duplicate shard content")
        seen_shard_sha.add(digest)
        if vocab_size is not None and maximum >= vocab_size:
            problems.append(
                f"{shard['file']}: token id {maximum} outside vocab {vocab_size}"
            )

        split = "val" if shard["file"].startswith("val_") else "train"
        doc_count = 0
        local_hashes: Counter[str] = Counter()
        with open(provenance, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                doc_count += 1
                value = record.get("sha256")
                if not value:
                    problems.append(f"{shard['file']}: provenance without sha256")
                    continue
                local_hashes[value] += 1
                split_hashes[split].add(value)
                if value in all_hashes and all_hashes[value] != path.name:
                    problems.append(
                        f"{shard['file']}: hash {value[:12]} also in {all_hashes[value]}"
                    )
                all_hashes[value] = path.name
        duplicates = sum(count - 1 for count in local_hashes.values() if count > 1)
        if duplicates:
            problems.append(f"{shard['file']}: {duplicates} duplicate provenance hashes")
        if doc_count != int(shard["documents"]):
            problems.append(
                f"{shard['file']}: docs {doc_count} != manifest {shard['documents']}"
            )

        total_tokens += token_count
        total_docs += doc_count
        shard_audits.append(
            {
                "file": shard["file"],
                "split": split,
                "tokens": token_count,
                "documents": doc_count,
                "size_bytes": size,
                "sha256": digest,
                "min_token_id": minimum,
                "max_token_id": maximum,
                "provenance_duplicates": duplicates,
            }
        )

    train_val_overlap = split_hashes["train"] & split_hashes["val"]
    if train_val_overlap:
        problems.append(f"{len(train_val_overlap)} hashes appear in both train and val")

    stats = manifest.get("stats", {})
    if total_tokens != int(stats.get("tokens_train", 0)) + int(stats.get("tokens_val", 0)):
        problems.append(
            f"total tokens {total_tokens} != stats "
            f"{int(stats.get('tokens_train', 0)) + int(stats.get('tokens_val', 0))}"
        )
    if total_docs != int(stats.get("records_train", 0)) + int(stats.get("records_val", 0)):
        problems.append(
            f"total docs {total_docs} != stats "
            f"{int(stats.get('records_train', 0)) + int(stats.get('records_val', 0))}"
        )

    result = {
        "dataset_dir": str(dataset_dir),
        "dataset": manifest.get("dataset"),
        "created_at": manifest.get("created_at"),
        "shards": shard_audits,
        "totals": {
            "tokens": total_tokens,
            "documents": total_docs,
            "unique_hashes": len(all_hashes),
            "train_hashes": len(split_hashes["train"]),
            "val_hashes": len(split_hashes["val"]),
        },
        "train_val_overlap": len(train_val_overlap),
        "problems": problems,
        "passed": not problems,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    vocab_size = args.vocab_size
    manifest_path = Path(args.dataset_dir) / "dataset_manifest.json"
    if vocab_size is None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vocab_size = manifest.get("tokenizer", {}).get("vocab_size")

    result = verify(Path(args.dataset_dir), vocab_size)
    print(json.dumps(result["totals"], indent=2))
    if result["passed"]:
        print("VERIFIED: shard sizes, SHA-256, token ranges, provenance, and splits are consistent.")
    else:
        print("FAILED:")
        for problem in result["problems"]:
            print(f"  - {problem}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"JSON written to {args.json_out}")

    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
