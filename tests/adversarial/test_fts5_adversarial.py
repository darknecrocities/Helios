"""Tier-5 Adversarial Stress Test Suite for SQLite FTS5 Hybrid Knowledge Base.

Adversarially probes:
1. Extreme punctuation and malformed queries:
   ?, !, :, extreme quotes, +, -, *, ^, (), [], {}, <>, /, \\, unbalanced brackets,
   and punctuation soup.
2. Boolean keywords and malformed operators:
   AND, OR, NOT, AND AND OR, NOT NOT, and chained prefix/postfix operator sequences.
3. SQL injection resistance:
   ' OR '1'='1, \"; DROP TABLE corpus_fts; --, UNION SELECT, table deletion/modification attempts.
4. Boundary and extreme length inputs:
   Single-character queries, whitespace-only, and 10,000-50,000 character payloads.
5. Unicode, emoji, accents, and Tagalog-English bilingual queries:
   Diacritics (José Rizal), non-Latin scripts, emojis, and code-switched queries.
6. High-concurrency multi-threading stress:
   20 concurrent threads hammering query_corpus() simultaneously, asserting
   zero sqlite3 database locked errors and latency <10ms per query.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import string
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from helios.memory.fts_corpus import (
    LOCAL_FALLBACK_DB,
    PRIMARY_EXTERNAL_DB,
    FTSCorpus,
    sanitize_fts5_query,
)


@pytest.fixture
def populated_corpus(tmp_path: Path) -> FTSCorpus:
    """Create a populated FTSCorpus in a temporary directory for isolated stress testing."""
    db_path = tmp_path / "adversarial_corpus.db"
    corpus = FTSCorpus(db_path=db_path)
    records: list[dict[str, Any]] = [
        {
            "id": f"doc_{i}",
            "domain": "science" if i % 2 == 0 else "history",
            "source": "adversarial_test",
            "title": f"Document Title {i}",
            "content": (
                f"Document {i} discusses photosynthesis, chlorophyll, solar cellular energy, "
                f"and historical events like Julius Caesar assassination in 44 BC."
            ),
        }
        for i in range(200)
    ]
    records.append(
        {
            "id": "doc_rizal",
            "domain": "tagalog",
            "source": "ph_history",
            "title": "Talambuhay ni Dr. José Rizal",
            "content": "Si Dr. José Rizal ay ipinanganak sa Calamba, Laguna noong Hunyo 19, 1861.",
        }
    )
    records.append(
        {
            "id": "doc_med",
            "domain": "medicine",
            "source": "pharma",
            "title": "Glaucoma Beta Blocker",
            "content": "Betaxolol is a cardioselective beta-1 adrenergic receptor antagonist used in glaucoma.",
        }
    )
    corpus.index_batch(records)
    return corpus


# ==============================================================================
# 1. Extreme Punctuation and Malformed Queries
# ==============================================================================


@pytest.mark.tier5
class TestFTS5PunctuationAndMalformedQueries:
    """Adversarially probe FTS5 query parser with punctuation and malformed strings."""

    @pytest.mark.parametrize(
        "punct",
        [
            "?",
            "!",
            ":",
            '"""""',
            "'''",
            "+",
            "-",
            "*",
            "^",
            "()",
            "[]",
            "{}",
            "<>",
            "/",
            "\\",
            "??!!::++--**^^(())[[]]{{}}<<>>//\\\\",
            string.punctuation,
            '"""',
            "'''",
            "(((",
            ")))",
            "[[[",
            "]]]",
            "{{{",
            "}}}",
            "/?><;:'][{}=-+_)(*&^%$#@!~`",
        ],
    )
    def test_pure_punctuation_does_not_crash(self, populated_corpus: FTSCorpus, punct: str) -> None:
        """Verify queries consisting solely of punctuation never raise sqlite3 syntax errors."""
        sanitized = sanitize_fts5_query(punct)
        assert isinstance(sanitized, str)
        hits = populated_corpus.query_corpus(punct)
        assert isinstance(hits, list)

    @pytest.mark.parametrize(
        "malformed",
        [
            'photosynthesis: "unclosed string',
            'Julius (Caesar OR "assassination',
            "Betaxolol +++ --- *** ^^^",
            "<<<photosynthesis>>>",
            "[[[chlorophyll]]] AND ???",
            '"""photosynthesis"""',
            "'''44 BC'''",
            "Dr. José Rizal :::: (Calamba)",
        ],
    )
    def test_mixed_punctuation_retains_terms_without_crash(
        self, populated_corpus: FTSCorpus, malformed: str
    ) -> None:
        """Verify queries with unbalanced punctuation extract word tokens and execute safely."""
        hits = populated_corpus.query_corpus(malformed, top_k=3)
        assert isinstance(hits, list)
        assert len(hits) >= 1
        assert hits[0].score > 0.0

    @pytest.mark.parametrize(
        "non_matching_malformed",
        [
            r"C:\Windows\System32\drivers\etc\hosts",
            "/usr/local/bin/python3 -m pip install",
            "+++ --- *** ^^^ ??? !!!",
        ],
    )
    def test_non_matching_malformed_queries_return_empty_safely(
        self, populated_corpus: FTSCorpus, non_matching_malformed: str
    ) -> None:
        """Verify malformed queries for non-existent content return empty list without crash."""
        hits = populated_corpus.query_corpus(non_matching_malformed, top_k=3)
        assert isinstance(hits, list)
        assert len(hits) == 0


# ==============================================================================
# 2. Boolean Keywords Hardening
# ==============================================================================


@pytest.mark.tier5
class TestFTS5BooleanKeywordsHardening:
    """Verify boolean operator keywords in various invalid positions do not corrupt FTS5 syntax."""

    @pytest.mark.parametrize(
        "query",
        [
            "AND",
            "OR",
            "NOT",
            "AND AND OR",
            "NOT NOT",
            "AND OR NOT",
            "OR OR OR",
            "NOT AND",
            "AND NOT OR AND",
            "AND photosynthesis",
            "photosynthesis OR",
            "NOT Julius Caesar",
            "photosynthesis AND OR NOT Caesar",
            "photosynthesis AND AND chlorophyll",
            "NOT NOT Betaxolol",
            "AND OR NOT AND OR NOT photosynthesis",
        ],
    )
    def test_boolean_keywords_quoted_and_safe(self, populated_corpus: FTSCorpus, query: str) -> None:
        """Verify operators are treated as literals or sanitized into safe quoted terms."""
        sanitized = sanitize_fts5_query(query)
        # Operators must be wrapped in double quotes
        if "AND" in query:
            assert '"AND"' in sanitized or "and" in sanitized.lower()
        hits = populated_corpus.query_corpus(query, top_k=2)
        assert isinstance(hits, list)
        if "photosynthesis" in query.lower() or "caesar" in query.lower() or "betaxolol" in query.lower():
            assert len(hits) >= 1


# ==============================================================================
# 3. SQL Injection Resistance
# ==============================================================================


@pytest.mark.tier5
class TestFTS5SQLInjectionResistance:
    """Verify SQL injection vectors are defanged by parameterization and sanitization."""

    @pytest.mark.parametrize(
        "sqli_payload",
        [
            "' OR '1'='1",
            '"; DROP TABLE corpus_fts; --',
            "'; DELETE FROM corpus_fts WHERE 1=1; --",
            "1' UNION SELECT id, domain, source, title, content, 99.0 FROM corpus_fts --",
            "' OR ''='",
            "admin'--",
            "'; VACUUM; --",
            "'; ATTACH DATABASE ':memory:' AS evil; --",
            "'; UPDATE corpus_fts SET content='hacked'; --",
            "') OR 1=1 --",
            "\" OR \"\"=\"",
        ],
    )
    def test_sqli_payloads_do_not_alter_database_or_leak(
        self, populated_corpus: FTSCorpus, sqli_payload: str
    ) -> None:
        """Verify malicious SQL payloads are harmlessly queried or sanitized to empty."""
        count_before = populated_corpus.count_records()
        hits = populated_corpus.query_corpus(sqli_payload)
        count_after = populated_corpus.count_records()

        # Schema and records must remain completely intact
        assert count_after == count_before
        assert count_after > 0
        assert isinstance(hits, list)


# ==============================================================================
# 4. Boundary and Extreme Length Queries
# ==============================================================================


@pytest.mark.tier5
class TestFTS5BoundaryAndLengthQueries:
    """Adversarially probe single-character, blank, and extreme-length query strings."""

    @pytest.mark.parametrize("char", ["a", "z", "0", "9", "_", "!", " ", "\t", "\n", "   \n\t  "])
    def test_single_character_and_blank_queries(
        self, populated_corpus: FTSCorpus, char: str
    ) -> None:
        """Verify single character or blank queries execute safely without error."""
        hits = populated_corpus.query_corpus(char)
        assert isinstance(hits, list)

    def test_query_exceeding_10000_characters_repeated_token(
        self, populated_corpus: FTSCorpus
    ) -> None:
        """Verify 15,000-character repetitive query executes within SLA without recursion/crash."""
        long_query = "photosynthesis " * 1000  # 15,000 characters
        assert len(long_query) > 10000
        t0 = time.perf_counter()
        hits = populated_corpus.query_corpus(long_query, top_k=3)
        latency_ms = (time.perf_counter() - t0) * 1000

        assert len(hits) >= 1
        assert hits[0].score > 0.0
        assert latency_ms < 2000.0  # Safe execution under extreme adversarial 15k-char payload

    def test_query_with_thousands_of_distinct_tokens(self, populated_corpus: FTSCorpus) -> None:
        """Verify 2,000 distinct tokens in a query (>16,000 chars) are handled without error."""
        distinct_long = " ".join(f"token_{i}" for i in range(2000))
        assert len(distinct_long) > 16000
        hits = populated_corpus.query_corpus(distinct_long, top_k=2)
        assert isinstance(hits, list)
        assert len(hits) == 0


# ==============================================================================
# 5. Unicode, Emojis, and Bilingual Queries
# ==============================================================================


@pytest.mark.tier5
class TestFTS5UnicodeEmojiAndBilingualQueries:
    """Adversarially probe unicode accents, emojis, and Tagalog-English bilingual queries."""

    @pytest.mark.parametrize(
        "query,expected_keyword",
        [
            ("Dr. José Rizal Calamba Laguna", "rizal"),
            ("José", "rizal"),
            ("Laguna noong Hunyo 1861", "rizal"),
            ("🔥⚡🌱 photosynthesis chlorophyll", "photosynthesis"),
            ("🇵🇭 Maynila at Laguna 🇵🇭", "rizal"),
            ("café résumé façade", ""),
            ("日本語 中文 العربية", ""),
            ("Kamusta world! Anong lagay ng photosynthesis?", "photosynthesis"),
            ("Saan matatagpuan si José Rizal sa Calamba?", "rizal"),
        ],
    )
    def test_unicode_and_bilingual_retrieval(
        self, populated_corpus: FTSCorpus, query: str, expected_keyword: str
    ) -> None:
        """Verify unicode accents and mixed Tagalog-English inputs retrieve matching docs."""
        hits = populated_corpus.query_corpus(query, top_k=2)
        assert isinstance(hits, list)
        if expected_keyword:
            assert len(hits) >= 1
            assert any(expected_keyword in h.content.lower() for h in hits)
            assert hits[0].score > 0.0


# ==============================================================================
# 6. High-Concurrency Stress Test (20 Concurrent Threads)
# ==============================================================================


@pytest.mark.tier5
class TestFTS5HighConcurrencyStress:
    """Hammer FTSCorpus with 20 concurrent threads asserting 0 locked errors and <10ms latency."""

    def test_20_threads_concurrency_stress_temporary_corpus(
        self, populated_corpus: FTSCorpus
    ) -> None:
        """Verify 20 concurrent threads execute 400 queries with 0 locked errors and avg <10ms."""
        num_threads = 20
        queries_per_thread = 20  # 400 total queries
        queries = [
            "photosynthesis chlorophyll",
            "Julius Caesar 44 BC",
            "Dr. José Rizal Calamba",
            "Betaxolol glaucoma antagonist",
            "cellular solar energy",
        ]

        latencies: list[float] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            populated_corpus.query_corpus("warmup", top_k=1)
            for i in range(queries_per_thread):
                q = queries[(tid + i) % len(queries)]
                t0 = time.perf_counter()
                try:
                    res = populated_corpus.query_corpus(q, top_k=2)
                    lat = (time.perf_counter() - t0) * 1000
                    with lock:
                        latencies.append(lat)
                        if len(res) == 0:
                            errors.append(RuntimeError(f"Thread {tid} got 0 hits for {q}"))
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        t_start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_time = time.perf_counter() - t_start

        # Assert zero errors (zero sqlite3.OperationalError: database is locked)
        assert len(errors) == 0, f"Encountered concurrency errors: {errors[:5]}"
        assert len(latencies) == num_threads * queries_per_thread

        avg_latency = sum(latencies) / len(latencies)
        sorted_lat = sorted(latencies)
        p95_latency = sorted_lat[int(len(latencies) * 0.95)]
        max_latency = max(latencies)

        # Assert performance metrics under 20-thread concurrency
        assert avg_latency < 10.0, f"Avg latency {avg_latency:.2f}ms exceeded 10ms SLA"
        assert p95_latency < 35.0, f"P95 latency {p95_latency:.2f}ms exceeded 35ms tail threshold"
        assert max_latency < 150.0, f"Max latency {max_latency:.2f}ms exceeded 150ms ceiling"
        assert total_time < 3.0, f"Total execution {total_time:.2f}s exceeded 3.0s ceiling"

    def test_20_threads_concurrency_stress_active_corpus(self) -> None:
        """Verify 20 concurrent threads against active 134k corpus if present on disk."""
        active_db = LOCAL_FALLBACK_DB if LOCAL_FALLBACK_DB.exists() else PRIMARY_EXTERNAL_DB
        if not active_db.exists():
            pytest.skip("Active corpus DB not found on disk")

        corpus = FTSCorpus(db_path=active_db)
        assert corpus.count_records() >= 1000

        target_queries = [
            "Julius Caesar",
            "photosynthesis",
            "Betaxolol",
            "Arron Kian Parejas",
            "Calamba Laguna",
        ]

        # Warm-up thread-local connections
        for q in target_queries:
            corpus.query_corpus(q, top_k=2)

        num_threads = 20
        queries_per_thread = 10  # 200 total queries
        latencies: list[float] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(tid: int) -> None:
            # Initialize thread-local connection and WAL pragmas
            corpus.query_corpus("warmup", top_k=1)
            for i in range(queries_per_thread):
                q = target_queries[(tid + i) % len(target_queries)]
                t0 = time.perf_counter()
                try:
                    res = corpus.query_corpus(q, top_k=2)
                    lat = (time.perf_counter() - t0) * 1000
                    with lock:
                        latencies.append(lat)
                        if len(res) == 0:
                            errors.append(RuntimeError(f"Zero hits for {q}"))
                except Exception as exc:
                    with lock:
                        errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Zero locked errors invariant
        assert len(errors) == 0, f"Encountered database locked or execution errors: {errors[:5]}"
        assert len(latencies) == num_threads * queries_per_thread

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 10.0, f"Avg latency {avg_latency:.2f}ms exceeded 10ms on warm DB"
        corpus.close()
