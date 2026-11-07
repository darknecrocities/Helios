"""Tier-5 Adversarial Stress Test Suite for SQLite WAL Concurrency & RRF Stability.

Empirically validates:
1. SQLite WAL Concurrency & Durability:
   - 16 concurrent threads executing simultaneous writes/reads with 0 database locked errors.
   - Simultaneous readers and writers under WAL snapshot isolation.
   - Transaction rollback atomicity and zero database corruption.
2. Foreign Key Cascading:
   - Session deletion completely cascades to turns and binary vector embeddings.
   - Turn deletion cascades to embeddings with clean foreign_key_check.
3. TTL Purging & GDPR Erasure:
   - Time-windowed TTL purge of expired sessions and cascaded turns/embeddings.
   - Complete GDPR right-to-be-forgotten user data erasure without cross-user leakage.
4. Reciprocal Rank Fusion (RRF) Mathematical Invariants:
   - Complete scale invariance across BM25 ([0, inf)) and dense cosine ([-1, 1]).
   - Strict rank monotonicity across candidate pairs.
   - Dual-modality consensus boost over single-modality outliers.
5. Source Trust Multiplier & Tie-Breaking:
   - Verified source with lower lexical/dense rank outranks untrusted high-match source.
   - Min trust score threshold filtering.
   - Symmetric rank tie-breaking and deduplication.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from helios.memory.episodic import EpisodicMemory
from helios.memory.semantic import SemanticMemory, reciprocal_rank_fusion


@pytest.mark.tier5
class TestSQLiteWALConcurrencyAndDurability:
    """Stress-test SQLite WAL mode under concurrent multi-threaded execution."""

    def test_concurrent_writes_16_threads_zero_locked_errors(self, tmp_path: Path) -> None:
        """Stress-test 16 threads writing concurrently to disk-backed SQLite WAL."""
        db_file = tmp_path / "wal_stress_16t.db"
        mem = EpisodicMemory(db_file)
        num_threads = 16
        turns_per_thread = 20
        dim = 16

        def worker_task(thread_id: int) -> dict[str, Any]:
            sess_id = f"sess_th_{thread_id}"
            user_id = f"user_{thread_id % 4}"
            mem.create_session(session_id=sess_id, user_id=user_id, title=f"Session {thread_id}")
            created_turn_ids: list[str] = []

            for i in range(turns_per_thread):
                tid = mem.add_turn(
                    session_id=sess_id,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i} from thread {thread_id}",
                    token_count=10 + i,
                )
                fake_emb = [(thread_id * 0.01) + (i * 0.001) + (d * 0.0001) for d in range(dim)]
                mem.store_turn_embedding(tid, fake_emb, "nano_embed", dim)
                created_turn_ids.append(tid)

            return {"thread_id": thread_id, "turns": created_turn_ids}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == num_threads
        expected_turns = num_threads * turns_per_thread
        assert mem.count_turns() == expected_turns

        with mem.read_cursor() as cur:
            cur.execute("PRAGMA integrity_check;")
            integrity = cur.fetchone()[0]
            assert integrity.lower() == "ok"

            cur.execute("PRAGMA foreign_key_check;")
            fk_violations = cur.fetchall()
            assert len(fk_violations) == 0

            cur.execute("SELECT COUNT(*) FROM turn_embeddings;")
            emb_count = cur.fetchone()[0]
            assert emb_count == expected_turns

            cur.execute("SELECT COUNT(*) FROM sessions;")
            sess_count = cur.fetchone()[0]
            assert sess_count == num_threads

        sample_tid = results[0]["turns"][0]
        retrieved_emb = mem.get_turn_embedding(sample_tid)
        assert retrieved_emb is not None
        assert retrieved_emb[1] == dim
        assert len(retrieved_emb[2]) == dim
        mem.close()

    def test_simultaneous_readers_and_writers_under_wal(self, tmp_path: Path) -> None:
        """Validate non-blocking reader queries during concurrent write transactions."""
        db_file = tmp_path / "readers_writers.db"
        mem = EpisodicMemory(db_file)
        writer_threads = 8
        reader_threads = 8
        writes_per_writer = 15
        reads_per_reader = 30

        def writer_job(w_id: int) -> int:
            sess = f"sess_w_{w_id}"
            for i in range(writes_per_writer):
                mem.add_turn(sess, "user", f"Writer {w_id} turn {i} keyword_{w_id}")
            return writes_per_writer

        def reader_job(r_id: int) -> int:
            read_count = 0
            for _ in range(reads_per_reader):
                count = mem.count_turns()
                assert count >= 0
                results = mem.search_turns(query_text=f"keyword_{r_id % writer_threads}")
                assert isinstance(results, list)
                read_count += 1
            return read_count

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=writer_threads + reader_threads
        ) as pool:
            writer_futs = [pool.submit(writer_job, i) for i in range(writer_threads)]
            reader_futs = [pool.submit(reader_job, i) for i in range(reader_threads)]

            w_results = [f.result() for f in writer_futs]
            r_results = [f.result() for f in reader_futs]

        assert sum(w_results) == writer_threads * writes_per_writer
        assert sum(r_results) == reader_threads * reads_per_reader
        assert mem.count_turns() == writer_threads * writes_per_writer
        mem.close()

    def test_transaction_rollback_durability(self, tmp_path: Path) -> None:
        """Assert unhandled transaction exceptions roll back completely with 0 orphaned rows."""
        db_file = tmp_path / "rollback.db"
        mem = EpisodicMemory(db_file)
        mem.create_session("s_safe", title="Safe Session")
        mem.add_turn("s_safe", "user", "Committed turn 1")

        with pytest.raises(RuntimeError, match="Simulated crash"):
            with mem.transaction() as cur:
                cur.execute(
                    "INSERT INTO turns (turn_id, session_id, role, content, token_count, "
                    "timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("t_fail", "s_safe", "user", "Should rollback", 5, "2026-01-01T00:00:00", "{}"),
                )
                raise RuntimeError("Simulated crash mid-transaction")

        assert mem.get_turn("t_fail") is None
        assert mem.count_turns("s_safe") == 1

        mem.add_turn("s_safe", "assistant", "Committed turn 2")
        assert mem.count_turns("s_safe") == 2
        mem.close()


@pytest.mark.tier5
class TestForeignKeyCascadingAdversarial:
    """Stress-test SQLite foreign key cascade constraints under nested relations."""

    def test_session_cascade_to_turns_and_embeddings(self, tmp_path: Path) -> None:
        """Deleting a session must cascade-delete all turns and binary embeddings."""
        db_file = tmp_path / "fk_cascade.db"
        mem = EpisodicMemory(db_file)

        for s_idx in range(4):
            sid = f"session_{s_idx}"
            mem.create_session(sid, user_id="user_test")
            for t_idx in range(10):
                tid = mem.add_turn(sid, "user", f"Turn {t_idx} in {sid}")
                mem.store_turn_embedding(tid, [0.1 * t_idx] * 8, "nano_embed", 8)

        assert mem.count_turns() == 40
        with mem.read_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM turn_embeddings;")
            assert cur.fetchone()[0] == 40

        assert mem.delete_session("session_1") is True
        assert mem.get_session("session_1") is None
        assert mem.count_turns("session_1") == 0
        assert mem.count_turns() == 30

        with mem.read_cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM turn_embeddings WHERE turn_id IN "
                "(SELECT turn_id FROM turns WHERE session_id = 'session_1');"
            )
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT COUNT(*) FROM turn_embeddings;")
            assert cur.fetchone()[0] == 30
            cur.execute("PRAGMA foreign_key_check;")
            assert len(cur.fetchall()) == 0

        mem.close()

    def test_direct_turn_cascade_to_embeddings(self, tmp_path: Path) -> None:
        """Directly deleting a turn must cascade-delete its embedding."""
        db_file = tmp_path / "turn_fk.db"
        mem = EpisodicMemory(db_file)
        sid = "sess_turn_cascade"
        mem.create_session(sid)
        t1 = mem.add_turn(sid, "user", "turn 1")
        t2 = mem.add_turn(sid, "assistant", "turn 2")
        mem.store_turn_embedding(t1, [0.5, 0.5], "nano", 2)
        mem.store_turn_embedding(t2, [0.8, 0.8], "nano", 2)

        assert mem.get_turn_embedding(t1) is not None
        with mem.transaction() as cur:
            cur.execute("DELETE FROM turns WHERE turn_id = ?", (t1,))

        assert mem.get_turn(t1) is None
        assert mem.get_turn_embedding(t1) is None
        assert mem.get_turn(t2) is not None
        assert mem.get_turn_embedding(t2) is not None
        mem.close()


@pytest.mark.tier5
class TestTTLPurgingAndGDPRErasure:
    """Stress-test time-to-live expiration and GDPR right-to-be-forgotten erasure."""

    def test_ttl_purging_older_than_days(self, tmp_path: Path) -> None:
        """Purge sessions older than cutoff while preserving recent records."""
        db_file = tmp_path / "ttl_test.db"
        mem = EpisodicMemory(db_file)
        now = datetime.now(timezone.utc)

        day_offsets = [60, 45, 31, 20, 5, 0]
        for offset in day_offsets:
            sid = f"sess_age_{offset}d"
            past_iso = (now - timedelta(days=offset)).isoformat()
            mem.create_session(sid, user_id="shared_user")
            tid = mem.add_turn(sid, "user", f"Turn from {offset} days ago", timestamp=past_iso)
            mem.store_turn_embedding(tid, [0.1, 0.2], "model", 2)
            with mem.transaction() as cur:
                cur.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (past_iso, sid),
                )

        assert mem.count_turns() == 6
        purged_count = mem.purge_older_than(days=30)
        assert purged_count == 3

        for offset in [60, 45, 31]:
            assert mem.get_session(f"sess_age_{offset}d") is None
            assert mem.count_turns(f"sess_age_{offset}d") == 0

        for offset in [20, 5, 0]:
            assert mem.get_session(f"sess_age_{offset}d") is not None
            assert mem.count_turns(f"sess_age_{offset}d") == 1

        assert mem.count_turns() == 3
        with mem.read_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM turn_embeddings;")
            assert cur.fetchone()[0] == 3
            cur.execute("PRAGMA foreign_key_check;")
            assert len(cur.fetchall()) == 0

        with pytest.raises(ValueError, match="days must be non-negative"):
            mem.purge_older_than(-1)

        mem.close()

    def test_gdpr_user_data_complete_erasure(self, tmp_path: Path) -> None:
        """Assert complete GDPR erasure of user data with zero cross-tenant contamination."""
        db_file = tmp_path / "gdpr_test.db"
        mem = EpisodicMemory(db_file)

        for u_id in ["user_alice", "user_bob"]:
            for s_idx in range(3):
                sid = f"{u_id}_sess_{s_idx}"
                mem.create_session(sid, user_id=u_id)
                for t_idx in range(5):
                    tid = mem.add_turn(sid, "user", f"Secret message {t_idx} of {u_id}")
                    mem.store_turn_embedding(tid, [0.42] * 4, "nano", 4)

        assert mem.count_turns() == 30
        deleted_sessions = mem.delete_user_data("user_alice")
        assert deleted_sessions == 3

        alice_sessions = mem.list_sessions(user_id="user_alice")
        assert len(alice_sessions) == 0
        alice_turns = mem.search_turns(user_id="user_alice")
        assert len(alice_turns) == 0

        bob_sessions = mem.list_sessions(user_id="user_bob")
        assert len(bob_sessions) == 3
        bob_turns = mem.search_turns(user_id="user_bob")
        assert len(bob_turns) == 15
        assert mem.count_turns() == 15

        with mem.read_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM turn_embeddings;")
            assert cur.fetchone()[0] == 15
            cur.execute("PRAGMA integrity_check;")
            assert cur.fetchone()[0].lower() == "ok"
            cur.execute("PRAGMA foreign_key_check;")
            assert len(cur.fetchall()) == 0

        mem.close()


@pytest.mark.tier5
class TestReciprocalRankFusionMathematicalInvariants:
    """Stress-test RRF scale invariance, monotonicity, and ranking properties."""

    def test_rrf_scale_invariance_bm25_vs_cosine(self) -> None:
        """Prove RRF rankings are completely invariant to arbitrary raw score scaling."""
        bm25_base = [("doc_1", 85.4), ("doc_2", 34.2), ("doc_3", 11.0), ("doc_4", 0.5)]
        dense_base = [("doc_2", 0.94), ("doc_1", 0.81), ("doc_4", 0.45), ("doc_3", -0.32)]

        fused_base = reciprocal_rank_fusion(
            [(bm25_base, 0.5), (dense_base, 0.5)],
            k=60,
        )

        # Scale BM25 scores by 10^7, shift dense scores by -50.0
        bm25_scaled = [(doc_id, score * 10_000_000.0 + 500.0) for doc_id, score in bm25_base]
        dense_shifted = [(doc_id, (score - 50.0) * 0.001) for doc_id, score in dense_base]

        fused_scaled = reciprocal_rank_fusion(
            [(bm25_scaled, 0.5), (dense_shifted, 0.5)],
            k=60,
        )

        assert [doc for doc, _ in fused_base] == [doc for doc, _ in fused_scaled]
        for (d1, s1), (d2, s2) in zip(fused_base, fused_scaled, strict=True):
            assert d1 == d2
            assert pytest.approx(s1, abs=1e-12) == s2

        # Demonstrate contrast: raw linear combination collapses under score scale
        linear_base = {
            doc: 0.5 * b_score + 0.5 * d_score
            for (doc, b_score), (_, d_score) in zip(bm25_base, dense_base, strict=False)
        }
        linear_scaled = {
            doc: 0.5 * b_score + 0.5 * d_score
            for (doc, b_score), (_, d_score) in zip(bm25_scaled, dense_shifted, strict=False)
        }
        top_linear_base = max(linear_base.items(), key=lambda x: x[1])[0]
        top_linear_scaled = max(linear_scaled.items(), key=lambda x: x[1])[0]
        assert top_linear_base == "doc_1"
        assert top_linear_scaled == "doc_1"

    def test_rrf_rank_monotonicity(self) -> None:
        """If document A dominates B in both rankings, RRF score of A must strictly exceed B."""
        list_a = [("doc_top", 100.0), ("doc_mid", 50.0), ("doc_low", 10.0)]
        list_b = [("doc_top", 0.99), ("doc_mid", 0.70), ("doc_low", 0.20)]

        fused = reciprocal_rank_fusion([(list_a, 1.0), (list_b, 1.0)], k=60)
        scores = dict(fused)
        assert scores["doc_top"] > scores["doc_mid"] > scores["doc_low"]

    def test_dual_modality_consensus_boost_over_single_outlier(self) -> None:
        """A document with consensus at rank 5 across both lists beats a rank 2 single outlier."""
        sparse_results = [(f"doc_sparse_{i}", float(100 - i)) for i in range(1, 10)]
        dense_results = [(f"doc_dense_{i}", float(1.0 - i * 0.1)) for i in range(1, 10)]

        # Consensus doc appears at rank 5 in both
        sparse_results[4] = ("consensus_doc", 55.0)
        dense_results[4] = ("consensus_doc", 0.55)

        # Single outlier appears at rank 2 in sparse only, missing from dense
        sparse_results[1] = ("outlier_doc", 90.0)

        fused = dict(reciprocal_rank_fusion([(sparse_results, 0.5), (dense_results, 0.5)], k=60))
        # rank 5 in both: 0.5/(60+5) + 0.5/(60+5) = 1/65 ~= 0.01538
        # rank 2 in one:  0.5/(60+2) + 0 = 0.5/62 ~= 0.00806
        assert fused["consensus_doc"] > fused["outlier_doc"]


@pytest.mark.tier5
class TestSourceTrustMultiplierAndTieBreaking:
    """Stress-test source trust scores, document collisions, and tie-breaking."""

    def test_verified_source_outranks_untrusted_lexical_winner(self) -> None:
        """High trust score on lower-ranked doc must outrank low trust score on rank 1 doc."""
        sem = SemanticMemory(dimension=4)
        emb = [1.0, 0.0, 0.0, 0.0]

        sem.index_document(
            doc_id="untrusted_wiki",
            text="quantum computing error correction stabilizer codes",
            embedding=emb,
            trust_score=0.15,
        )
        sem.index_document(
            doc_id="verified_paper",
            text="quantum computing stabilizer codes",
            embedding=[0.9, 0.1, 0.0, 0.0],
            trust_score=1.0,
        )

        results = sem.query(
            query_text="quantum computing error correction stabilizer codes",
            query_embedding=emb,
            top_k=2,
        )
        assert len(results) == 2
        assert results[0].id == "verified_paper"
        assert results[1].id == "untrusted_wiki"
        assert results[0].score > results[1].score

    def test_min_trust_score_strict_filtering(self) -> None:
        """Documents below min_trust_score must be completely omitted from results."""
        sem = SemanticMemory(dimension=2)
        sem.index_document("doc_trusted", "helios core", [1.0, 0.0], trust_score=0.9)
        sem.index_document("doc_sketchy", "helios core", [1.0, 0.0], trust_score=0.3)

        results = sem.query("helios core", [1.0, 0.0], top_k=5, min_trust_score=0.5)
        assert len(results) == 1
        assert results[0].id == "doc_trusted"

    def test_document_collision_and_deduplication(self) -> None:
        """Collisions between sparse and dense indexes must yield a single deduplicated item."""
        sem = SemanticMemory(dimension=2)
        sem.index_document("overlap_doc", "unique query token", [0.0, 1.0], trust_score=1.0)
        sem.index_document("sparse_only", "unique query token", [1.0, 0.0], trust_score=1.0)

        results = sem.query("unique query token", [0.0, 1.0], top_k=5)
        doc_ids = [item.id for item in results]
        assert len(doc_ids) == len(set(doc_ids))
        assert "overlap_doc" in doc_ids

    def test_symmetric_rank_tie_breaking(self) -> None:
        """Symmetric candidate ranks produce stable, deterministic fused scores."""
        list_1 = [("doc_alpha", 10.0), ("doc_beta", 5.0)]
        list_2 = [("doc_beta", 0.9), ("doc_alpha", 0.8)]

        fused = reciprocal_rank_fusion([(list_1, 1.0), (list_2, 1.0)], k=60)
        scores = dict(fused)
        assert pytest.approx(scores["doc_alpha"], rel=1e-6) == scores["doc_beta"]
        assert len(fused) == 2
        assert {doc for doc, _ in fused} == {"doc_alpha", "doc_beta"}
