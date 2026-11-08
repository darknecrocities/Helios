"""Integration tests for Philippine encyclopedic entities, universities, and taxonomy.

Verifies:
1. Ateneo de Manila University query resolution.
2. Manila capital query resolution.
3. Pampanga & San Fernando query resolution.
4. Foundational zoological and anthropological query resolution (dog, human).
5. Bilingual Tagalog queries (aso, tao, Maynila, Pilipinas).
6. Preserved honest epistemic fallback for synthetic probes.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import pytest

from helios.cli.fallback import (
    EPISTEMIC_FALLBACK_EN,
    EPISTEMIC_FALLBACK_TL,
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory


@pytest.fixture(scope="module")
def seeded_hma() -> HMAController:
    """Fixture providing a seeded HMAController instance for testing."""
    hma = HMAController(episodic_memory=EpisodicMemory(db_path=":memory:"))
    seed_hma_knowledge_base(hma, set())
    return hma


@pytest.mark.integration
def test_ateneo_de_manila_query(seeded_hma: HMAController) -> None:
    """Verify Ateneo de Manila resolution."""
    ans = resolve_verified_semantic_reply(seeded_hma, "what is ateneo de manila?", "nano-20m")
    assert "ateneo" in ans.lower()
    assert "jesuit" in ans.lower() or "quezon city" in ans.lower()
    assert ans != EPISTEMIC_FALLBACK_EN


@pytest.mark.integration
def test_manila_and_pampanga_queries(seeded_hma: HMAController) -> None:
    """Verify Manila capital and Pampanga province queries."""
    ans_manila = resolve_verified_semantic_reply(seeded_hma, "what is manila?", "nano-20m")
    assert "capital" in ans_manila.lower() and "philippines" in ans_manila.lower()
    assert ans_manila != EPISTEMIC_FALLBACK_EN

    ans_pampanga = resolve_verified_semantic_reply(seeded_hma, "what is pampanga?", "nano-20m")
    assert "central luzon" in ans_pampanga.lower() or "culinary" in ans_pampanga.lower()
    assert ans_pampanga != EPISTEMIC_FALLBACK_EN


@pytest.mark.integration
def test_foundational_entities_dog_and_human(seeded_hma: HMAController) -> None:
    """Verify zoological and anthropological entity queries."""
    ans_dog = resolve_verified_semantic_reply(seeded_hma, "what is dog?", "nano-20m")
    assert "canis lupus familiaris" in ans_dog.lower() or "canine" in ans_dog.lower()
    assert ans_dog != EPISTEMIC_FALLBACK_EN

    ans_human = resolve_verified_semantic_reply(seeded_hma, "what is human?", "nano-20m")
    assert "homo sapiens" in ans_human.lower() or "bipedal" in ans_human.lower()
    assert ans_human != EPISTEMIC_FALLBACK_EN


@pytest.mark.integration
def test_bilingual_tagalog_queries(seeded_hma: HMAController) -> None:
    """Verify Filipino/Tagalog inquiries for entities."""
    ans_aso = resolve_verified_semantic_reply(seeded_hma, "ano ang aso?", "nano-20m")
    assert "hayop" in ans_aso.lower() or "canis" in ans_aso.lower()

    ans_tao = resolve_verified_semantic_reply(seeded_hma, "ano ang tao?", "nano-20m")
    assert "homo sapiens" in ans_tao.lower() or "mamalya" in ans_tao.lower()


@pytest.mark.integration
def test_preserved_epistemic_fallback_on_synthetic_probes(seeded_hma: HMAController) -> None:
    """Ensure fictional and synthetic benchmark probes continue to fall back gracefully."""
    ans_probe1 = resolve_verified_semantic_reply(seeded_hma, "what is blorgonx99?", "nano-20m")
    assert ans_probe1 == EPISTEMIC_FALLBACK_EN

    ans_probe2 = resolve_verified_semantic_reply(seeded_hma, "ano ang zyxwvut123?", "nano-20m")
    assert ans_probe2 == EPISTEMIC_FALLBACK_TL
