"""SQLite FTS5 Hybrid Knowledge Base for Massive Pretraining Corpus Grounding.

Provides sub-10ms full-text retrieval across 134,728+ records using SQLite FTS5
with Porter stemming, unicode61 tokenization, thread-safe connection pooling,
and BM25 ranking.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("helios.memory.fts_corpus")

PRIMARY_EXTERNAL_DB = Path("/Volumes/One Touch/Helios/data/hma_full_corpus.db")
LOCAL_FALLBACK_DB = Path("data/hma_full_corpus.db")

PRIMARY_EXTERNAL_DATASET = Path(
    "/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl"
)
LOCAL_FALLBACK_DATASETS = [
    Path("data/helios_expanded_reasoning_corpus.jsonl"),
    Path("data/helios_contextual_reasoning.jsonl"),
]


@dataclass
class FTSDocument:
    """Document retrieved from FTS5 corpus index."""

    id: str
    domain: str
    source: str
    title: str
    content: str
    score: float = 0.0


def resolve_corpus_db_path(custom_path: str | Path | None = None) -> Path | str:
    """Resolve database path preferring external storage with local fallback."""
    if custom_path is not None:
        if str(custom_path) == ":memory:":
            return ":memory:"
        return Path(custom_path).resolve()

    if PRIMARY_EXTERNAL_DB.parent.exists():
        return PRIMARY_EXTERNAL_DB

    local_resolved = LOCAL_FALLBACK_DB.resolve()
    if local_resolved.exists():
        return local_resolved

    return local_resolved


def resolve_dataset_jsonl_path(custom_path: str | Path | None = None) -> Path:
    """Resolve path to JSONL corpus file."""
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified dataset file not found: {p}")

    if PRIMARY_EXTERNAL_DATASET.exists():
        return PRIMARY_EXTERNAL_DATASET

    for candidate in LOCAL_FALLBACK_DATASETS:
        res = candidate.resolve()
        if res.exists():
            return res

    return PRIMARY_EXTERNAL_DATASET


def sanitize_fts5_query(query: str) -> str:
    """Sanitize user query for SQLite FTS5 MATCH clause.

    Extracts word tokens, wraps each in double quotes to prevent syntax errors
    from punctuation or boolean operators, and joins with OR for maximum recall.
    """
    tokens = re.findall(r"\b\w+\b", query)
    if not tokens:
        return ""
    # Wrap in quotes to avoid syntax crashes on punctuation or operators (AND, OR, NOT)
    return " OR ".join(f'"{t}"' for t in tokens)


class FTSCorpus:
    """On-Disk SQLite FTS5 Full-Text Knowledge Base for Helios HMA."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        resolved = resolve_corpus_db_path(db_path)
        self.db_path = resolved
        self._is_memory = str(resolved) == ":memory:"
        self._local = threading.local()
        self._shared_mem_conn: sqlite3.Connection | None = None
        self._write_lock = threading.RLock()

        if self._is_memory:
            # For :memory:, keep a persistent connection so tables persist across threads/calls
            self._shared_mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._configure_connection(self._shared_mem_conn)

        self._init_schema()

    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply high-performance pragmas."""
        conn.row_factory = sqlite3.Row
        if not self._is_memory:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA busy_timeout = 10000;")
            conn.execute("PRAGMA mmap_size = 268435456;")
            conn.execute("PRAGMA cache_size = -64000;")

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection for disk DB or shared for :memory:."""
        if self._is_memory and self._shared_mem_conn is not None:
            return self._shared_mem_conn

        if not hasattr(self._local, "conn") or self._local.conn is None:
            p = str(self.db_path)
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(p, timeout=30.0)
            self._configure_connection(conn)
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize the virtual FTS5 table."""
        with self._write_lock:
            conn = self._get_connection()
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts USING fts5(
                    id UNINDEXED,
                    domain,
                    source,
                    title,
                    content,
                    tokenize = 'porter unicode61'
                );
                """
            )
            conn.commit()

    def index_record(
        self,
        doc_id: str,
        content: str,
        domain: str = "general",
        source: str = "dataset",
        title: str = "",
    ) -> None:
        """Index a single document into the FTS5 table."""
        with self._write_lock:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO corpus_fts(id, domain, source, title, content)
                VALUES (?, ?, ?, ?, ?);
                """,
                (doc_id, domain, source, title, content),
            )
            conn.commit()

    def index_batch(self, records: list[dict[str, Any]]) -> int:
        """Insert records in a single high-speed transaction."""
        if not records:
            return 0
        with self._write_lock:
            conn = self._get_connection()
            conn.execute("BEGIN TRANSACTION;")
            try:
                for r in records:
                    doc_id = str(r.get("id", ""))
                    domain = str(r.get("domain", "general"))
                    source = str(r.get("source", "pretrain"))
                    title = str(r.get("title", ""))
                    content = str(r.get("content", ""))
                    conn.execute(
                        """
                        INSERT INTO corpus_fts(id, domain, source, title, content)
                        VALUES (?, ?, ?, ?, ?);
                        """,
                        (doc_id, domain, source, title, content),
                    )
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
        return len(records)

    def build_from_jsonl(
        self,
        jsonl_path: str | Path | None = None,
        batch_size: int = 5000,
        max_records: int | None = None,
    ) -> int:
        """High-speed batch ingestion of JSONL corpus records with 5,000-row transactions."""
        path = resolve_dataset_jsonl_path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"JSONL corpus file does not exist: {path}")

        total_inserted = 0
        batch: list[dict[str, Any]] = []

        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                prompt = data.get("prompt", "").strip()
                response = data.get("response", "").strip()
                text = data.get("text", "").strip()

                if prompt and response:
                    content = f"Q: {prompt}\nA: {response}"
                    title = prompt[:120]
                elif text:
                    content = text
                    title = text[:120].replace("\n", " ")
                else:
                    continue

                doc_id = str(data.get("id") or f"doc_{idx}")
                domain = str(data.get("domain", "general"))
                source = str(data.get("source", path.stem))

                batch.append(
                    {
                        "id": doc_id,
                        "domain": domain,
                        "source": source,
                        "title": title,
                        "content": content,
                    }
                )

                if len(batch) >= batch_size:
                    self.index_batch(batch)
                    total_inserted += len(batch)
                    batch.clear()

                if max_records is not None and total_inserted >= max_records:
                    break

        if batch and (max_records is None or total_inserted < max_records):
            if max_records is not None:
                batch = batch[: max_records - total_inserted]
            self.index_batch(batch)
            total_inserted += len(batch)

        logger.info("Indexed %d records into FTS5 corpus (%s)", total_inserted, self.db_path)
        return total_inserted

    def query_corpus(
        self,
        query: str,
        top_k: int = 5,
        domain: str | None = None,
    ) -> list[FTSDocument]:
        """Execute BM25 search returning top_k documents with positive relevance scores."""
        if top_k <= 0:
            return []

        match_clause = sanitize_fts5_query(query)
        if not match_clause:
            return []

        conn = self._get_connection()
        if domain:
            sql = """
                SELECT id, domain, source, title, content, -bm25(corpus_fts) AS score
                FROM corpus_fts
                WHERE corpus_fts MATCH ? AND domain = ?
                ORDER BY score DESC
                LIMIT ?;
            """
            cursor = conn.execute(sql, (match_clause, domain, top_k))
        else:
            sql = """
                SELECT id, domain, source, title, content, -bm25(corpus_fts) AS score
                FROM corpus_fts
                WHERE corpus_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?;
            """
            cursor = conn.execute(sql, (match_clause, top_k))

        results: list[FTSDocument] = []
        for row in cursor.fetchall():
            results.append(
                FTSDocument(
                    id=str(row["id"]),
                    domain=str(row["domain"]),
                    source=str(row["source"]),
                    title=str(row["title"]),
                    content=str(row["content"]),
                    score=float(row["score"]),
                )
            )
        return results

    def count_records(self) -> int:
        """Return total number of records indexed in FTS5 table."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT count(*) FROM corpus_fts;")
        row = cursor.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close connections."""
        if self._is_memory and self._shared_mem_conn is not None:
            try:
                self._shared_mem_conn.close()
            except Exception:
                pass
            self._shared_mem_conn = None

        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None
