"""Cryptographic-hash deduplication and topic tracking for Colab pretraining."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("helios.colab_dedup")

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Coding": ("def ", "class ", "import ", "python", "algorithm", "function", "code", "typescript", "rust", "complexity"),
    "Math": ("theorem", "proof", "solve", "equation", "matrix", "integral", "derivative", "prime", "modulo", "algebra"),
    "Agents": ("tool", "mcp", "agent", "function_call", "jsonrpc", "schema", "execute", "plan", "thought", "workflow"),
    "Science": ("biology", "chemistry", "physics", "cell", "protein", "dna", "quantum", "molecule", "reaction", "energy"),
    "History": ("dynasty", "century", "war", "emperor", "treaty", "revolution", "president", "ancient", "medieval", "battle"),
    "Geography": ("capital", "mountain", "river", "ocean", "continent", "border", "country", "latitude", "longitude", "city"),
    "Tagalog": ("ang", "mga", "ng", "sa", "ay", "ito", "siya", "natin", "kanila", "pilipinas", "manila", "tagalog"),
    "Law": ("constitution", "statute", "jurisdiction", "court", "verdict", "plaintiff", "defendant", "legal", "treaty"),
}


def classify_record_topics(record: dict[str, Any]) -> list[str]:
    """Classifies a corpus record into one or more topic domains."""
    meta = " ".join(str(record.get(k, "")) for k in ("domain", "category", "source", "id") if record.get(k)).lower()
    p = str(record.get("prompt", ""))
    r = str(record.get("response", ""))
    t = str(record.get("text", ""))
    combined = f"{meta} {p} {r} {t}".lower()
    matched = [cat for cat, kws in TOPIC_KEYWORDS.items() if any(kw in combined for kw in kws)]
    return matched or ["General Web"]


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
    slice_topics: Counter[str] = Counter()

    while len(selected) < slice_size and len(seen_hashes_set) < total_records:
        rec = records[idx % total_records]
        r_hash = compute_record_hash(rec)
        if r_hash not in seen_hashes_set:
            selected.append(rec)
            seen_hashes_set.add(r_hash)
            ledger.setdefault("seen_hashes", []).append(r_hash)
            for top in classify_record_topics(rec):
                slice_topics[top] += 1
        else:
            skipped_dups += 1
        idx += 1
        if idx % total_records == 0:
            current_epoch = ledger.get("category_epochs", {}).get("all", 1)
            ledger.setdefault("category_epochs", {})["all"] = current_epoch + 1
            logger.info("Curriculum looped to Epoch %d. Seen unique hashes: %d", current_epoch + 1, len(seen_hashes_set))

    new_cursor = idx % total_records
    ledger.setdefault("category_cursors", {})["all"] = new_cursor
    
    # Update cumulative topics & unique counts
    cum_topics = Counter(ledger.get("cumulative_topics", {}))
    cum_topics.update(slice_topics)
    ledger["cumulative_topics"] = dict(cum_topics.most_common())
    ledger["cumulative_unique_records"] = int(ledger.get("cumulative_unique_records", 0)) + len(selected)
    
    total_t = sum(slice_topics.values())
    summary_parts = [f"{k} {int((v/total_t)*100)}%" for k, v in slice_topics.most_common(5)] if total_t else []
    summary_str = " · ".join(summary_parts) if summary_parts else "General Web 100%"
    ledger["latest_topics_summary"] = summary_str
    ledger["latest_cursor_range"] = f"Rec #{cursor:,}→#{new_cursor:,} (0% Dup)"

    session_entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "runner": "colab_gpu",
        "start_offset": cursor,
        "end_offset": new_cursor,
        "unique_records_sampled": len(selected),
        "duplicates_skipped": skipped_dups,
        "topics_breakdown": dict(slice_topics.most_common()),
        "trained_topics_summary": summary_str,
    }
    ledger.setdefault("sessions", []).append(session_entry)

    logger.info("Extracted %d unique records (skipped %d duplicates). New cursor=%d | Total unique seen=%d | Topics: %s",
                len(selected), skipped_dups, new_cursor, len(seen_hashes_set), summary_str)
    return selected, cursor, new_cursor
