#!/usr/bin/env python3
"""Build and index massive pretraining corpus into on-disk SQLite FTS5 database.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helios.memory.fts_corpus import (  # noqa: E402
    LOCAL_FALLBACK_DB,
    FTSCorpus,
    resolve_corpus_db_path,
    resolve_dataset_jsonl_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build high-performance SQLite FTS5 full-text corpus from JSONL."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to source JSONL dataset (defaults to external drive or local fallback).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to output SQLite FTS5 database (defaults to external drive or local fallback).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Transaction batch size for batch insertions (default: 5000).",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum records to index (useful for fast verification).",
    )
    parser.add_argument(
        "--mirror-local",
        action="store_true",
        default=True,
        help="Mirror built database to local data/hma_full_corpus.db for offline resilience.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = resolve_dataset_jsonl_path(args.source)
    db_path = resolve_corpus_db_path(args.db)

    print("=" * 70)
    print("HELIOS FTS5 CORPUS BUILDER")
    print("=" * 70)
    print(f"Source JSONL:  {source_path}")
    print(f"Target DB:     {db_path}")
    print(f"Batch Size:    {args.batch_size}")
    if args.max_records:
        print(f"Max Records:   {args.max_records}")
    print("=" * 70)

    start_time = time.perf_counter()
    corpus = FTSCorpus(db_path=db_path)

    total_indexed = corpus.build_from_jsonl(
        jsonl_path=source_path,
        batch_size=args.batch_size,
        max_records=args.max_records,
    )
    elapsed = time.perf_counter() - start_time
    rate = total_indexed / elapsed if elapsed > 0 else 0.0

    db_file = Path(str(db_path))
    size_mb = db_file.stat().st_size / (1024 * 1024) if db_file.exists() else 0.0

    print("-" * 70)
    print(f"Ingestion Complete: {total_indexed:,} records in {elapsed:.2f}s ({rate:,.1f} rec/s)")
    print(f"Database File Size: {size_mb:.2f} MB ({db_path})")

    # Sample query verification
    test_queries = [
        "Julius Caesar",
        "photosynthesis",
        "Arron Kian Parejas",
    ]
    print("-" * 70)
    print("Running Verification Queries:")
    for q in test_queries:
        t0 = time.perf_counter()
        hits = corpus.query_corpus(q, top_k=2)
        q_time_ms = (time.perf_counter() - t0) * 1000
        first_title = hits[0].title if hits else "NO MATCH"
        print(f"  Query: '{q}' -> {len(hits)} hits ({q_time_ms:.2f}ms) | Top: {first_title[:60]}")

    # Mirror to local directory if built on external drive
    if args.mirror_local and db_file != LOCAL_FALLBACK_DB.resolve():
        local_db = LOCAL_FALLBACK_DB.resolve()
        local_db.parent.mkdir(parents=True, exist_ok=True)
        print("-" * 70)
        print(f"Mirroring database to local fallback: {local_db}")
        shutil.copy2(db_file, local_db)
        # Also copy WAL and SHM if present
        for ext in ["-wal", "-shm"]:
            sidecar = db_file.with_name(db_file.name + ext)
            if sidecar.exists():
                shutil.copy2(sidecar, local_db.with_name(local_db.name + ext))
        local_size = local_db.stat().st_size / (1024 * 1024)
        print(f"Mirroring Complete: {local_size:.2f} MB")

    corpus.close()
    print("=" * 70)
    print("FTS5 Knowledge Base build successfully finished.")


if __name__ == "__main__":
    main()
