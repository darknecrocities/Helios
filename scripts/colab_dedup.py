"""Cryptographic-hash deduplication for the Colab pretraining runner."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger("helios.colab_dedup")


def compute_record_hash(record: dict[str, Any]) -> str:
    """Computes deterministic 64-bit SHA hash for deduplication."""
    text = record.get("text") or record.get("content") or json.dumps(record, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def get_deduplicated_record_slice(
    records: list[dict[str, Any]],
    ledger: dict[str, Any],
    slice_size: int = 1000,
) -> tuple[list[dict[str, Any]], int, int]:
    """Retrieves next slice of records, strictly guaranteeing 0% duplication."""
    total_records = len(records)
    cursor = ledger.get("category_cursors", {}).get("all", 0)
    seen_hashes_set = set(ledger.get("seen_hashes", []))

    selected: list[dict[str, Any]] = []
    idx = cursor
    skipped_dups = 0

    while len(selected) < slice_size and len(seen_hashes_set) < total_records:
        rec = records[idx % total_records]
        r_hash = compute_record_hash(rec)
        if r_hash not in seen_hashes_set:
            selected.append(rec)
            seen_hashes_set.add(r_hash)
            ledger.setdefault("seen_hashes", []).append(r_hash)
        else:
            skipped_dups += 1
        idx += 1
        if idx % total_records == 0:
            current_epoch = ledger.get("category_epochs", {}).get("all", 1)
            ledger.setdefault("category_epochs", {})["all"] = current_epoch + 1
            logger.info("Curriculum looped to Epoch %d. Seen unique hashes: %d", current_epoch + 1, len(seen_hashes_set))

    new_cursor = idx % total_records
    ledger.setdefault("category_cursors", {})["all"] = new_cursor
    logger.info("Extracted %d unique records (skipped %d duplicates). New cursor=%d | Total unique seen=%d",
                len(selected), skipped_dups, new_cursor, len(seen_hashes_set))
    return selected, cursor, new_cursor
