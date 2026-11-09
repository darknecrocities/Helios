"""Regression tests for anti-hallucination retrieval, multi-turn WorkingMemory, dynamic CoT,
non-repeating curriculum ledger, trained-topics status telemetry, hardware governor, and
creator identity privacy.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from helios.cli.fallback import (
    resolve_identity_or_greeting,
    resolve_verified_semantic_reply,
    sanitize_output_privacy,
    seed_hma_knowledge_base,
)
from helios.memory.bm25 import extract_content_tokens, has_conflicting_numerals
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory
from helios.reasoning.cot_engine import (
    build_dynamic_cot_trace,
    resolve_conversational_followup,
)
from helios_pipeline.data.curriculum_ledger import (
    sample_non_repeating_corpus,
    update_trained_topics_docs,
)
from helios_pipeline.evaluation.health import inspect_tier_checkpoint
from helios_pipeline.npu.hardware_governor import (
    compute_hardware_shift_plan,
    pace_training_step,
)


def test_numeral_disambiguation_ww2_vs_ww1() -> None:
    """Verify 'what happend in world war 2?' never matches World War I."""
    q_tokens = extract_content_tokens("what happend in world war 2?")
    assert any(x in q_tokens for x in ("2", "ii", "ww2"))
    assert has_conflicting_numerals("what happend in world war 2?", "What caused World War I?")
    assert not has_conflicting_numerals(
        "what happend in world war 2?", "What happened in World War II?"
    )

    hma = HMAController(episodic_memory=EpisodicMemory(db_path=":memory:"))
    seed_hma_knowledge_base(hma, set())

    reply = resolve_verified_semantic_reply(hma, "what happend in world war 2?", preset="core-1b")
    assert "World War I " not in reply
    assert "Franz Ferdinand" not in reply
    assert any(
        k in reply for k in ("1939", "1945", "WW2", "World War II", "Second World War", "Allies")
    )


def test_multi_turn_code_followup_no_hammurabi() -> None:
    """Verify 'what is this code about?' after Python generation explains the code, not Hammurabi."""
    sample_code_turn = (
        "```python\n"
        "import numpy as np\n"
        "from sklearn.datasets import make_blobs\n"
        "from sklearn.mixture import GaussianMixture\n\n"
        "def generate_gmm_dataset(n_samples, n_features, n_components, random_state):\n"
        "    # Generate random data points\n"
        "    X, _ = make_blobs(n_samples=n_samples, n_features=n_features)\n"
        "    # Fit a Gaussian Mixture Model\n"
        "    gmm = GaussianMixture(n_components=n_components)\n"
        "    return X\n"
        "```\n"
        "The `generate_gmm_dataset` function generates synthetic data using a Gaussian Mixture Model."
    )
    history = [
        {"role": "user", "content": "generate me a sample python code"},
        {"role": "assistant", "content": sample_code_turn},
    ]

    direct_reply, thought, rewritten = resolve_conversational_followup(
        "what is this code about?", history
    )
    assert direct_reply is not None
    assert "Hammurabi" not in direct_reply
    assert "generate_gmm_dataset" in direct_reply
    assert "GaussianMixture" in direct_reply
    assert thought is not None and "Code of Hammurabi" in thought
    assert rewritten is None

    hma = HMAController(episodic_memory=EpisodicMemory(db_path=":memory:"))
    seed_hma_knowledge_base(hma, set())
    sem_reply = resolve_verified_semantic_reply(
        hma, "what is this code about?", preset="core-1b", history=history
    )
    assert "Hammurabi" not in sem_reply
    assert "generate_gmm_dataset" in sem_reply


def test_dynamic_cot_trace_is_context_aware() -> None:
    """Verify build_dynamic_cot_trace produces turn-specific reasoning rather than static boilerplate."""
    history = [
        {"role": "user", "content": "who is jose rizal?"},
        {"role": "assistant", "content": "Dr. Jose Rizal is the national hero of the Philippines."},
    ]
    trace = build_dynamic_cot_trace(
        query="what happend in world war 2?",
        reply="World War II lasted from 1939 to 1945.",
        history=history,
        source_label="HMA Hybrid Memory",
        preset="core-1b",
    )
    assert "Numeral & Entity Disambiguation" in trace
    assert "WW2 ≠ WWI" in trace
    assert "core-1b" in trace


def test_creator_identity_and_zero_email_disclosure() -> None:
    """Verify creator biography includes Domodomo, Top 1 on AppBuilders, GDG Lead, and 0 emails."""
    queries = [
        "who created you?",
        "who is arron kian parejas?",
        "what is domodomo?",
        "who is top 1 on appbuilders?",
        "who is the former chapter lead of gdg?",
        "who am i?",
        "what is arron kian parejas email address?",
    ]
    for q in queries:
        ans = resolve_identity_or_greeting(q, preset="core-1b")
        assert ans is not None, f"Expected identity reply for: {q}"
        assert "@" not in ans, f"Email leaked in reply to '{q}': {ans}"
        assert "gmail.com" not in ans.lower()
        assert "Arron Kian Parejas" in ans
        assert "Domodomo" in ans
        assert "AppBuilders" in ans
        assert "GDG" in ans or "Google Developer Groups" in ans

    redacted = sanitize_output_privacy(
        "Created by Arron Kian Parejas <parejasarronkian@gmail.com> (parejasarronkian@gmail.com)."
    )
    assert "@" not in redacted
    assert "gmail" not in redacted


def test_non_repeating_curriculum_ledger_zero_duplication(tmp_path: Path) -> None:
    """Verify consecutive sample_non_repeating_corpus runs advance cursor with 0% duplication."""

    class DummyStorage:
        def __init__(self, base: Path) -> None:
            self.base_dir = base

    base = tmp_path / "onetouch"
    ds_dir = base / "datasets"
    ds_dir.mkdir(parents=True)
    corpus_file = ds_dir / "helios_massive_pretrain_corpus.jsonl"

    rows = [
        json.dumps(
            {"prompt": f"Solve math equation {i}x = {i * 2}", "response": f"x = 2 (item {i})"}
        )
        for i in range(1, 120)
    ]
    corpus_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    storage = DummyStorage(base)

    # Initialize cursor at 0 for test
    manifests = base / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / "curriculum_ledger.json").write_text(
        json.dumps(
            {"category_cursors": {"all": 0}, "category_epochs": {"all": 1}, "seen_hashes": []}
        ),
        encoding="utf-8",
    )

    batch1, meta1 = sample_non_repeating_corpus(
        storage,
        limit=25,
        seed=1,
        category="all",
        advance_cursor=True,  # type: ignore[arg-type]
    )
    batch2, meta2 = sample_non_repeating_corpus(
        storage,
        limit=25,
        seed=1,
        category="all",
        advance_cursor=True,  # type: ignore[arg-type]
    )
    assert meta1["start_offset"] == 0
    assert meta2["start_offset"] == meta1["end_offset"]
    assert meta2["start_offset"] > 0
    # Verify non-overlapping corpus records (excluding the small CoT pool suffix)
    raw1 = {t for t in batch1 if "Solve math equation" in t}
    raw2 = {t for t in batch2 if "Solve math equation" in t}
    assert raw1.isdisjoint(raw2)

    doc_out = update_trained_topics_docs(
        json.loads((manifests / "curriculum_ledger.json").read_text(encoding="utf-8")),
        doc_path=tmp_path / "TRAINED_TOPICS_HISTORY.md",
    )
    assert doc_out.exists()
    assert "Zero-Duplication Guarantee" in doc_out.read_text(encoding="utf-8")


def test_hardware_governor_and_status_trained_topics() -> None:
    """Verify hardware governor shift plan and trained_topics_summary on checkpoint health."""
    plan = compute_hardware_shift_plan("nano", "cpu")
    assert plan.batch_size in (1, 2)
    assert plan.seq_len in (96, 128, 160)
    assert plan.cpu_threads >= 1
    pace_training_step(torch.device("cpu"), 1, plan)

    rep = inspect_tier_checkpoint("nano")
    assert rep.trained_topics_summary
    assert rep.cursor_range
