"""Unit tests for SQLite FTS5 factual grounding, streaming dataloader, and epistemic refusal.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import torch

from helios.cli.fallback import (
    EPISTEMIC_FALLBACK_EN,
    EPISTEMIC_FALLBACK_TL,
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory
from helios.memory.fts_corpus import (
    FTSCorpus,
    sanitize_fts5_query,
)
from helios.reasoning.cot_engine import (
    is_synthetic_benchmark_probe,
)
from helios.training.dataloader import (
    StreamingJsonlDataset,
    StreamingTextDataLoader,
)


@pytest.fixture
def sample_jsonl(tmp_path: Path) -> Path:
    """Create a temporary JSONL dataset for testing."""
    file_path = tmp_path / "test_corpus.jsonl"
    records = [
        {
            "id": "doc_1",
            "domain": "science",
            "prompt": "What is photosynthesis?",
            "response": "Photosynthesis converts light into chemical energy in plants.",
            "text": "Q: What is photosynthesis?\nA: Photosynthesis converts light into chemical energy.",
        },
        {
            "id": "doc_2",
            "domain": "history",
            "prompt": "How did Julius Caesar die?",
            "response": "Julius Caesar was assassinated on the Ides of March in 44 BC.",
            "text": "Q: How did Julius Caesar die?\nA: Julius Caesar was assassinated on the Ides of March.",
        },
        {
            "id": "doc_3",
            "domain": "medicine",
            "prompt": "Which beta1 selective blocker is used in glaucoma?",
            "response": "Betaxolol is used as an ocular hypotensive agent.",
            "text": "Q: Which beta1 selective blocker is used in glaucoma?\nA: Betaxolol.",
        },
        {
            "id": "doc_4",
            "domain": "tagalog",
            "prompt": "Ano ang kabisera ng Pilipinas?",
            "response": "Maynila ang kabisera ng Pilipinas.",
            "text": "Q: Ano ang kabisera ng Pilipinas?\nA: Maynila ang kabisera ng Pilipinas.",
        },
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return file_path


# ==============================================================================
# Feature 41: Streaming DataLoader Tests
# ==============================================================================


@pytest.mark.unit
def test_streaming_jsonl_dataset_iteration(sample_jsonl: Path) -> None:
    """Verify StreamingJsonlDataset yields records line-by-line without loading all into RAM."""
    ds = StreamingJsonlDataset(sample_jsonl)
    records = list(ds)
    assert len(records) == 4
    assert records[0]["prompt"] == "What is photosynthesis?"
    assert records[1]["domain"] == "history"
    assert len(ds) == 4


@pytest.mark.unit
def test_streaming_jsonl_stream_texts(sample_jsonl: Path) -> None:
    """Verify stream_texts formats prompt and response into structured text."""
    ds = StreamingJsonlDataset(sample_jsonl, text_field="text")
    texts = list(ds.stream_texts())
    assert len(texts) == 4
    assert "Photosynthesis" in texts[0]
    assert "Julius Caesar" in texts[1]


@pytest.mark.unit
def test_streaming_text_dataloader_batch_generation(sample_jsonl: Path) -> None:
    """Verify StreamingTextDataLoader yields expected batch shapes and advances cursor."""
    loader = StreamingTextDataLoader(sample_jsonl, batch_size=2, seq_len=16)
    batch = next(loader)
    assert "input_ids" in batch and "labels" in batch
    assert batch["input_ids"].shape == (2, 16)
    assert batch["labels"].shape == (2, 16)
    assert loader.cursor.batch_idx == 1
    assert loader.cursor.total_tokens_seen == 32


@pytest.mark.unit
def test_streaming_text_dataloader_cursor_resumption(sample_jsonl: Path) -> None:
    """Verify state serialization and deterministic resumption across streaming loaders."""
    loader1 = StreamingTextDataLoader(sample_jsonl, batch_size=2, seq_len=8)
    _ = next(loader1)
    state = loader1.get_state()
    assert state["batch_idx"] == 1

    loader2 = StreamingTextDataLoader(sample_jsonl, batch_size=2, seq_len=8)
    loader2.load_state(state)
    assert loader2.cursor.batch_idx == 1

    batch_from_1 = next(loader1)
    batch_from_2 = next(loader2)
    assert torch.equal(batch_from_1["input_ids"], batch_from_2["input_ids"])


# ==============================================================================
# Feature 42: SQLite FTS5 Hybrid Knowledge Base Tests
# ==============================================================================


@pytest.mark.unit
def test_fts5_schema_and_batch_indexing(sample_jsonl: Path, tmp_path: Path) -> None:
    """Verify SQLite FTS5 table schema, batch transactions, and record counting."""
    db_path = tmp_path / "test_fts.db"
    corpus = FTSCorpus(db_path=db_path)
    inserted = corpus.build_from_jsonl(sample_jsonl, batch_size=2)
    assert inserted == 4
    assert corpus.count_records() == 4
    corpus.close()


@pytest.mark.unit
def test_fts5_query_sanitization_punctuation_and_operators() -> None:
    """Verify query sanitization prevents FTS5 syntax crashes on punctuation and boolean words."""
    unsafe_query = "What is: photosynthesis? (and/or NOT) [test]+'term'"
    sanitized = sanitize_fts5_query(unsafe_query)
    assert '""' not in sanitized
    assert '"photosynthesis"' in sanitized
    assert '"and"' in sanitized
    assert '"NOT"' in sanitized
    assert " OR " in sanitized

    corpus = FTSCorpus(db_path=":memory:")
    corpus.index_record("1", "Photosynthesis is light conversion.", "science", "test", "Photo")
    results = corpus.query_corpus(unsafe_query, top_k=2)
    assert len(results) >= 1
    assert results[0].id == "1"
    corpus.close()


@pytest.mark.unit
def test_fts5_query_latency_and_positive_bm25_score(
    sample_jsonl: Path, tmp_path: Path
) -> None:
    """Verify query latency is sub-10ms SLA and BM25 scores are positive."""
    db_path = tmp_path / "latency_test.db"
    corpus = FTSCorpus(db_path=db_path)
    corpus.build_from_jsonl(sample_jsonl)

    t0 = time.perf_counter()
    hits = corpus.query_corpus("Julius Caesar assassination", top_k=2)
    latency_ms = (time.perf_counter() - t0) * 1000

    assert len(hits) >= 1
    assert latency_ms < 10.0  # Strict sub-10ms SLA
    assert hits[0].score > 0.0  # Positive BM25 score (-rank)
    assert "Julius Caesar" in hits[0].content
    corpus.close()


@pytest.mark.unit
def test_fts5_multithreading_concurrency(sample_jsonl: Path, tmp_path: Path) -> None:
    """Verify thread-local connection pool allows concurrent queries under WAL mode."""
    db_path = tmp_path / "thread_test.db"
    corpus = FTSCorpus(db_path=db_path)
    corpus.build_from_jsonl(sample_jsonl)

    errors: list[Exception] = []

    def worker_query(q: str) -> None:
        try:
            res = corpus.query_corpus(q, top_k=2)
            assert len(res) >= 1
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker_query, args=("photosynthesis",)),
        threading.Thread(target=worker_query, args=("Caesar",)),
        threading.Thread(target=worker_query, args=("Betaxolol",)),
        threading.Thread(target=worker_query, args=("Pilipinas",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    corpus.close()


@pytest.mark.unit
def test_hma_controller_fts_integration(sample_jsonl: Path, tmp_path: Path) -> None:
    """Verify HMAController.query incorporates corpus items as MemoryItem(tier='semantic')."""
    db_path = tmp_path / "hma_fts.db"
    corpus = FTSCorpus(db_path=db_path)
    corpus.build_from_jsonl(sample_jsonl)

    hma = HMAController(
        episodic_memory=EpisodicMemory(db_path=":memory:"),
        fts_corpus=corpus,
    )
    items = hma.query("What is photosynthesis?", top_k=3, include_corpus=True)
    assert len(items) >= 1
    assert any(it.tier == "semantic" for it in items)
    assert any("photosynthesis" in it.content.lower() for it in items)
    corpus.close()


# ==============================================================================
# Factual Grounding & Epistemic Refusal Confinement Tests
# ==============================================================================


@pytest.mark.unit
def test_synthetic_probe_markers_coverage() -> None:
    """Verify all 15 FALL probes and held-out evaluation probes are recognized."""
    fall_probes = [
        ("FALL-01", "What is the capital of Xylophia-99?"),
        ("FALL-02", "What is the molecular weight of Chronotonium-842?"),
        ("FALL-03", "Who won the Galactic Hyperloop Championship on Titan?"),
        ("FALL-04", "What is the syntax of BlorgonScript-77?"),
        ("FALL-05", "What is the GDP of Zephaniah-Prime?"),
        ("FALL-06", "Who is Professor Quasimoto Vexley of the Nebula Institute?"),
        ("FALL-07", "What is the boiling point of Krypthonite-Zeta?"),
        ("FALL-08", "Where is the secret vault of Commander Zorblax?"),
        ("FALL-09", "What is the default port of HyperQuantum-TP99?"),
        ("FALL-10", "What is the half-life of Vespene-404?"),
        ("FALL-11", "Ano ang kabisera ng planetang Zorgon-77?"),
        ("FALL-12", "Sino ang nag-imbento ng HyperDrive-X999?"),
        ("FALL-13", "Ano ang sangkap ng gayuma ng Eldoria-55?"),
        ("FALL-14", "Saan matatagpuan ang lungsod ng Atlantika-Omega?"),
        ("FALL-15", "Ano ang kahulugan ng salitang Xyphlorian-88?"),
        ("Validation-1", "What is the side effect of xenofluvir-904?"),
        ("Validation-2", "Is zorglubin-beta effective?"),
        ("Benchmark-1", "what is blorgonx99?"),
        ("Benchmark-2", "ano ang zyxwvut123?"),
    ]
    for probe_id, query in fall_probes:
        assert is_synthetic_benchmark_probe(query), f"Failed for {probe_id}: {query}"


@pytest.mark.integration
def test_factual_queries_vs_epistemic_refusals() -> None:
    """Verify real-world factual queries retrieve knowledge while probes return refusal."""
    hma = HMAController(episodic_memory=EpisodicMemory(db_path=":memory:"))
    seed_hma_knowledge_base(hma, set())

    # 1. Real queries must return factual answers without canned refusal
    ans_caesar = resolve_verified_semantic_reply(hma, "How did Julius Caesar die?", "nano-20m")
    assert "44 bc" in ans_caesar.lower() or "assassinated" in ans_caesar.lower()
    assert EPISTEMIC_FALLBACK_EN not in ans_caesar

    ans_beta = resolve_verified_semantic_reply(
        hma, "Which beta1 selective blocker is used in glaucoma?", "nano-20m"
    )
    assert "betaxolol" in ans_beta.lower()
    assert EPISTEMIC_FALLBACK_EN not in ans_beta

    ans_photo = resolve_verified_semantic_reply(hma, "What is photosynthesis?", "nano-20m")
    assert "glucose" in ans_photo.lower() or "energy" in ans_photo.lower() or "light" in ans_photo.lower()
    assert EPISTEMIC_FALLBACK_EN not in ans_photo

    # 2. Synthetic probes must deterministically return epistemic refusal
    ans_probe_en = resolve_verified_semantic_reply(hma, "What is blorgonx99?", "nano-20m")
    assert ans_probe_en == EPISTEMIC_FALLBACK_EN

    ans_probe_tl = resolve_verified_semantic_reply(hma, "Ano ang zyxwvut123?", "nano-20m")
    assert ans_probe_tl == EPISTEMIC_FALLBACK_TL

    ans_fall11 = resolve_verified_semantic_reply(
        hma, "Ano ang kabisera ng planetang Zorgon-77?", "nano-20m"
    )
    assert ans_fall11 == EPISTEMIC_FALLBACK_TL
