"""Persistent SQLite-backed episodic memory store with WAL mode for Helios."""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator

from helios.memory.base import VALID_ROLES, Session, Turn

SCHEMA_STATEMENTS: tuple[str, ...] = (
    (
        "CREATE TABLE IF NOT EXISTS sessions ("
        "session_id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'default_user', "
        "title TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
        "metadata TEXT NOT NULL DEFAULT '{}');"
    ),
    (
        "CREATE TABLE IF NOT EXISTS turns ("
        "turn_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL, "
        "content TEXT NOT NULL, token_count INTEGER NOT NULL DEFAULT 0, "
        "timestamp TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', "
        "FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE);"
    ),
    (
        "CREATE TABLE IF NOT EXISTS turn_embeddings ("
        "turn_id TEXT PRIMARY KEY, embedding_model TEXT NOT NULL, "
        "dimensions INTEGER NOT NULL, embedding_blob BLOB NOT NULL, "
        "created_at TEXT NOT NULL, "
        "FOREIGN KEY (turn_id) REFERENCES turns(turn_id) ON DELETE CASCADE);"
    ),
    ("CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC);"),
    ("CREATE INDEX IF NOT EXISTS idx_turns_session_timestamp ON turns(session_id, timestamp ASC);"),
    "CREATE INDEX IF NOT EXISTS idx_turns_role ON turns(role);",
    "CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns(timestamp ASC);",
)


class EpisodicMemory:
    """Thread-safe, WAL-enabled persistent SQLite episodic memory store."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path: str = str(db_path)
        self.is_memory: bool = self.db_path == ":memory:"
        self._local: threading.local = threading.local()
        self._write_lock: threading.RLock = threading.RLock()
        self._master_conn: sqlite3.Connection | None = None

        if self.is_memory:
            self._master_conn = self._create_connection()

        self._initialize_schema()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL pragmas."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=10.0,
            check_same_thread=False if self.is_memory else True,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection or shared in-memory connection."""
        if self.is_memory:
            assert self._master_conn is not None
            return self._master_conn
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager yielding cursor within an atomic write transaction."""
        with self._write_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    @contextmanager
    def read_cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager yielding cursor for read operations."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _initialize_schema(self) -> None:
        """Execute DDL statements to construct tables and indexes."""
        with self.transaction() as cur:
            for stmt in SCHEMA_STATEMENTS:
                cur.execute(stmt)

    def close(self) -> None:
        """Close thread connection and master connection if present."""
        with self._write_lock:
            if hasattr(self._local, "conn") and self._local.conn is not None:
                self._local.conn.close()
                self._local.conn = None
            if self._master_conn is not None:
                self._master_conn.close()
                self._master_conn = None

    def __enter__(self) -> EpisodicMemory:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def create_session(
        self,
        session_id: str | None = None,
        user_id: str = "default_user",
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Create a new conversational session record."""
        sid = session_id or f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{id(self)}"
        now = datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {})

        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO sessions (
                    session_id, user_id, title, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sid, user_id, title, now, now, meta_str),
            )
        return Session(
            session_id=sid,
            user_id=user_id,
            title=title,
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve session record by ID."""
        with self.read_cursor() as cur:
            cur.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            row = cur.fetchone()
            if not row:
                return None
            return Session(
                session_id=row["session_id"],
                user_id=row["user_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]),
            )

    def list_sessions(
        self, user_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[Session]:
        """List sessions ordered by update recency."""
        with self.read_cursor() as cur:
            if user_id is not None:
                cur.execute(
                    "SELECT * FROM sessions WHERE user_id = ? "
                    "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (user_id, limit, offset),
                )
            else:
                cur.execute(
                    "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            return [
                Session(
                    session_id=r["session_id"],
                    user_id=r["user_id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    metadata=json.loads(r["metadata"]),
                )
                for r in cur.fetchall()
            ]

    def delete_session(self, session_id: str) -> bool:
        """Delete session and cascade delete all associated turns and embeddings."""
        with self.transaction() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return cur.rowcount > 0

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        token_count: int = 0,
        metadata: dict[str, Any] | None = None,
        turn_id: str | None = None,
        timestamp: str | None = None,
    ) -> str:
        """Persist dialogue turn and update session updated_at timestamp."""
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}")

        tid = turn_id or f"turn_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        meta_str = json.dumps(metadata or {})

        with self.transaction() as cur:
            cur.execute(
                """
                INSERT OR IGNORE INTO sessions (
                    session_id, user_id, title, created_at, updated_at, metadata
                ) VALUES (?, 'default_user', NULL, ?, ?, '{}')
                """,
                (session_id, ts, ts),
            )
            cur.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, role, content, token_count, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (tid, session_id, role, content, token_count, ts, meta_str),
            )
            cur.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (ts, session_id),
            )
        return tid

    store_turn = add_turn

    def get_turn(self, turn_id: str) -> Turn | None:
        """Retrieve single turn by ID."""
        with self.read_cursor() as cur:
            cur.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,))
            row = cur.fetchone()
            if not row:
                return None
            return Turn(
                turn_id=row["turn_id"],
                role=row["role"],
                content=row["content"],
                token_count=row["token_count"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
            )

    def get_session_turns(self, session_id: str, limit: int | None = None) -> list[Turn]:
        """Fetch all turns for a session in chronological order."""
        with self.read_cursor() as cur:
            if limit is not None:
                cur.execute(
                    "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
                    (session_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM turns WHERE session_id = ? ORDER BY timestamp ASC",
                    (session_id,),
                )
            return [
                Turn(
                    turn_id=r["turn_id"],
                    role=r["role"],
                    content=r["content"],
                    token_count=r["token_count"],
                    timestamp=r["timestamp"],
                    metadata=json.loads(r["metadata"]),
                )
                for r in cur.fetchall()
            ]

    def count_turns(self, session_id: str | None = None) -> int:
        """Count turns across all sessions or within a specific session."""
        with self.read_cursor() as cur:
            if session_id is not None:
                cur.execute("SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM turns")
            return int(cur.fetchone()[0])

    def search_turns(
        self,
        session_id: str | None = None,
        user_id: str | None = None,
        role: str | None = None,
        start_time: str | datetime | None = None,
        end_time: str | datetime | None = None,
        query_text: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Turn]:
        """Multi-criteria search over interaction turns with parameterized SQL."""
        conditions: list[str] = []
        params: list[Any] = []

        if user_id is not None:
            conditions.append("t.session_id IN (SELECT session_id FROM sessions WHERE user_id = ?)")
            params.append(user_id)

        if session_id is not None:
            conditions.append("t.session_id = ?")
            params.append(session_id)

        if role is not None:
            conditions.append("t.role = ?")
            params.append(role)

        if start_time is not None:
            start_iso = (
                start_time.isoformat() if isinstance(start_time, datetime) else str(start_time)
            )
            conditions.append("t.timestamp >= ?")
            params.append(start_iso)

        if end_time is not None:
            end_iso = end_time.isoformat() if isinstance(end_time, datetime) else str(end_time)
            conditions.append("t.timestamp <= ?")
            params.append(end_iso)

        if query_text is not None and query_text.strip():
            conditions.append("t.content LIKE ?")
            params.append(f"%{query_text.strip()}%")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT t.* FROM turns t
            {where_clause}
            ORDER BY t.timestamp ASC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        with self.read_cursor() as cur:
            cur.execute(sql, params)
            return [
                Turn(
                    turn_id=r["turn_id"],
                    role=r["role"],
                    content=r["content"],
                    token_count=r["token_count"],
                    timestamp=r["timestamp"],
                    metadata=json.loads(r["metadata"]),
                )
                for r in cur.fetchall()
            ]

    def purge_older_than(self, days: int, user_id: str | None = None) -> int:
        """Purge sessions and cascade-delete turns older than the specified TTL days."""
        if days < 0:
            raise ValueError(f"days must be non-negative, got {days}")
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        with self.transaction() as cur:
            if user_id is not None:
                cur.execute(
                    "DELETE FROM sessions WHERE updated_at < ? AND user_id = ?",
                    (cutoff, user_id),
                )
            else:
                cur.execute("DELETE FROM sessions WHERE updated_at < ?", (cutoff,))
            return cur.rowcount

    def prune_session(self, session_id: str, keep_last_n_turns: int) -> int:
        """Delete older turns in a session, retaining only the most recent N turns."""
        if keep_last_n_turns < 0:
            raise ValueError("keep_last_n_turns must be non-negative")

        with self.transaction() as cur:
            if keep_last_n_turns == 0:
                cur.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
                return cur.rowcount

            cur.execute(
                """
                DELETE FROM turns
                WHERE session_id = ? AND turn_id NOT IN (
                    SELECT turn_id FROM turns
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
                """,
                (session_id, session_id, keep_last_n_turns),
            )
            return cur.rowcount

    def delete_user_data(self, user_id: str) -> int:
        """Hard delete all sessions and turns belonging to a user (GDPR compliance)."""
        with self.transaction() as cur:
            cur.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return cur.rowcount

    def export_session(self, session_id: str) -> dict[str, Any]:
        """Export session metadata and chronological turns to structured JSON."""
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Session '{session_id}' not found.")
        turns = self.get_session_turns(session_id)
        return {
            "session": session.to_dict(),
            "turn_count": len(turns),
            "turns": [t.to_dict() for t in turns],
        }

    def store_turn_embedding(
        self,
        turn_id: str,
        embedding: list[float] | bytes,
        model_name: str,
        dimensions: int,
    ) -> None:
        """Store compact float32 binary vector embedding for a dialogue turn."""
        blob = (
            embedding
            if isinstance(embedding, bytes)
            else struct.pack(f"{len(embedding)}f", *embedding)
        )
        now = datetime.now(timezone.utc).isoformat()

        with self.transaction() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO turn_embeddings (
                    turn_id, embedding_model, dimensions, embedding_blob, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (turn_id, model_name, dimensions, blob, now),
            )

    def get_turn_embedding(self, turn_id: str) -> tuple[str, int, list[float]] | None:
        """Retrieve stored embedding unpacked to list of float32 values."""
        with self.read_cursor() as cur:
            cur.execute(
                "SELECT embedding_model, dimensions, embedding_blob "
                "FROM turn_embeddings WHERE turn_id = ?",
                (turn_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            dims = row["dimensions"]
            blob = row["embedding_blob"]
            unpacked = list(struct.unpack(f"{dims}f", blob))
            return (row["embedding_model"], dims, unpacked)
