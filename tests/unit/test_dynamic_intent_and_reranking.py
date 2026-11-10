"""Unit and integration tests for dynamic intent classification and Q-to-Q reranking.

Verifies:
1. Intent classification across Entity Definition, Mathematics, Programming, and General Factual.
2. Incompatible modality rejection (math solutions and code snippets filtered from entity queries).
3. Elimination of false-positive subset matching (e.g. Europe vs Gutenberg printing press).
4. Full conversational turn verification for Napoleon, Napoleon Bonaparte, Europe, and Donald Trump.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import pytest

from helios.cli.chat import ChatSession
from helios.cli.fallback import resolve_verified_semantic_reply
from helios.cli.utils import load_model_and_config, load_or_create_tokenizer
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory
from helios.reasoning.intent import (
    QueryIntent,
    classify_query_intent,
    clean_meta_prompt_answer,
    is_candidate_intent_compatible,
    score_qa_alignment,
)


@pytest.mark.unit
def test_classify_query_intent() -> None:
    """Verify accurate intent identification across modalities."""
    assert classify_query_intent("who is napoleon?") == QueryIntent.ENTITY_DEFINITION
    assert classify_query_intent("who is donald trump?") == QueryIntent.ENTITY_DEFINITION
    assert classify_query_intent("what is europe?") == QueryIntent.ENTITY_DEFINITION
    assert classify_query_intent("sino si jose rizal?") == QueryIntent.ENTITY_DEFINITION
    assert classify_query_intent("calculate 15 * 24 + 10") == QueryIntent.MATHEMATICS
    assert classify_query_intent("what is 25% of 80?") == QueryIntent.MATHEMATICS
    assert classify_query_intent("write a python script to parse a json") == QueryIntent.PROGRAMMING
    assert classify_query_intent("create a function in python to reverse a list") == QueryIntent.PROGRAMMING


@pytest.mark.unit
def test_modality_filtering_rejects_incompatible_candidates() -> None:
    """Verify that entity queries reject math calculations, python scripts, and subparts."""
    # Reject math problem calculation for biographical query
    math_cand = "If Sedrich has 4 more jelly beans than Napoleon... #### 19"
    assert not is_candidate_intent_compatible(
        QueryIntent.ENTITY_DEFINITION,
        cand_prompt="If Napoleon has 17 jelly beans...",
        cand_resp=math_cand,
        query="who is napoleon?",
    )

    # Reject python code for biographical query
    code_cand = "def print_dict(d):\n    for k, v in d.items(): print(k, v)"
    assert not is_candidate_intent_compatible(
        QueryIntent.ENTITY_DEFINITION,
        cand_prompt="Create a function in Python to print a dictionary",
        cand_resp=code_cand,
        query="who is donald trump?",
    )

    # Reject subpart attribute questions (e.g. tallest mountain in Europe for what is europe)
    assert not is_candidate_intent_compatible(
        QueryIntent.ENTITY_DEFINITION,
        cand_prompt="What is the tallest mountain in Europe?",
        cand_resp="Mount Elbrus is the tallest mountain in Europe.",
        query="what is europe?",
    )


@pytest.mark.unit
def test_clean_meta_prompt_answer() -> None:
    """Verify stripping of trivia question/answer wrappers."""
    raw = "Question: Who were the commanders at Waterloo?\nAnswer: Napoleon Bonaparte and Wellington"
    assert clean_meta_prompt_answer(raw) == "Napoleon Bonaparte and Wellington"


@pytest.mark.integration
def test_transcript_queries_resolution() -> None:
    """Verify end-to-end resolution of user transcript queries without mock or static data."""
    hma = HMAController(episodic_memory=EpisodicMemory(db_path=":memory:"))

    ans_napoleon = resolve_verified_semantic_reply(hma, "who is napoleon?", preset="nano-20m")
    assert "jelly beans" not in ans_napoleon.lower()
    assert "napoleon" in ans_napoleon.lower() and "military" in ans_napoleon.lower()

    ans_bonaparte = resolve_verified_semantic_reply(hma, "who is napoleon bonaparte?", preset="nano-20m")
    assert "question: who were" not in ans_bonaparte.lower()
    assert "french revolution" in ans_bonaparte.lower() or "corsica" in ans_bonaparte.lower()

    ans_europe = resolve_verified_semantic_reply(hma, "what is europe?", preset="nano-20m")
    assert "gutenberg" not in ans_europe.lower()
    assert "continent" in ans_europe.lower()

    ans_trump = resolve_verified_semantic_reply(hma, "who is donald trump?", preset="nano-20m")
    assert "def print_dict" not in ans_trump.lower()
    assert "president" in ans_trump.lower()
