"""Unit and integration tests for the 500-question benchmark and epistemic fallback.

Verifies:
1. All 500 benchmark questions have unique IDs and valid schemas.
2. BM25 stopword filtering prevents stopword-only false matches ("42" bug).
3. Subject-aware retrieval and epistemic fallback accurately answer terminal
   transcript queries and gracefully decline out-of-session queries in EN/TL.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import pytest

from helios.cli.fallback import (
    EPISTEMIC_FALLBACK_EN,
    EPISTEMIC_FALLBACK_TL,
    is_tagalog_text,
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.evaluation.benchmark_500 import ALL_500_QUESTIONS
from helios.memory.bm25 import BM25Okapi, extract_content_tokens
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory


@pytest.mark.unit
@pytest.mark.tier1
def test_benchmark_500_completeness_and_uniqueness() -> None:
    """Verify the 500-question benchmark contains 500 unique, valid items."""
    assert len(ALL_500_QUESTIONS) == 500
    ids = [q.id for q in ALL_500_QUESTIONS]
    assert len(set(ids)) == 500
    prompts = [q.prompt.strip().lower() for q in ALL_500_QUESTIONS]
    assert len(set(prompts)) == 500


@pytest.mark.unit
@pytest.mark.tier2
def test_bm25_stopword_filtering_prevents_42_bug() -> None:
    """Verify stopword-only overlap ('what is') scores 0.0 in BM25Okapi."""
    bm25 = BM25Okapi()
    bm25.add_documents(
        [
            ("doc1", "Question: What is 15 + 27?\nAnswer: 42"),
            ("doc2", "Question: What is the capital of France?\nAnswer: Paris"),
        ]
    )
    results = bm25.search("what is unknownxyzsubject?", top_k=2)
    assert results == []


@pytest.mark.unit
@pytest.mark.tier3
def test_epistemic_fallback_and_subject_verification() -> None:
    """Verify content token extraction and bilingual language detection."""
    assert extract_content_tokens("What is Google?") == ["google"]
    assert is_tagalog_text("Ano ang zyxwvut123?")
    assert not is_tagalog_text("What is blorgonx99?")


@pytest.mark.integration
@pytest.mark.tier4
def test_terminal_transcript_queries_and_fallback() -> None:
    """Verify all user-reported terminal queries and unseen fallback probes."""
    hma = HMAController(episodic_memory=EpisodicMemory(db_path=":memory:"))
    seed_hma_knowledge_base(hma, set())

    # Previously broken terminal transcript queries must return accurate answers
    ans_google = resolve_verified_semantic_reply(hma, "what is google?", "nano-20m")
    assert "search" in ans_google.lower() and ans_google != "42"

    ans_comp = resolve_verified_semantic_reply(hma, "what is computer?", "nano-20m")
    assert "electronic" in ans_comp.lower() and ans_comp != "42"

    ans_js = resolve_verified_semantic_reply(hma, "what is javascript?", "nano-20m")
    assert "programming" in ans_js.lower() and ans_js != "42"

    ans_luzon = resolve_verified_semantic_reply(hma, "ano ang luzon?", "nano-20m")
    assert "pinakamalaki" in ans_luzon.lower()

    ans_ver = resolve_verified_semantic_reply(hma, "what is model version?", "nano-20m")
    assert "v0.2.0" in ans_ver.lower()

    # Unseen out-of-session queries must return the honest epistemic fallback
    ans_unseen_en = resolve_verified_semantic_reply(hma, "what is blorgonx99?", "nano-20m")
    assert ans_unseen_en == EPISTEMIC_FALLBACK_EN

    ans_unseen_tl = resolve_verified_semantic_reply(hma, "ano ang zyxwvut123?", "nano-20m")
    assert ans_unseen_tl == EPISTEMIC_FALLBACK_TL
