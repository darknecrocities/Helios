"""Unit tests for the autonomous self-improving benchmark evaluation suite.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import pytest

from helios.training.auto_improve import BENCHMARK_SUITE, AutoImprover, BenchmarkQuestion


def test_benchmark_question_evaluate_positive():
    q = BenchmarkQuestion(
        id="test_1",
        category="identity",
        question="Who created Helios?",
        expected_answer="Arron Kian Parejas",
        keywords=["Arron", "Parejas"],
    )
    ok, reason = q.evaluate("Helios was created and developed by Arron Kian Parejas.")
    assert ok is True
    assert reason == "Passed"


def test_benchmark_question_evaluate_negative():
    q = BenchmarkQuestion(
        id="test_2",
        category="identity",
        question="Who created Helios?",
        expected_answer="Arron Kian Parejas",
        keywords=["Arron", "Parejas"],
        negative_keywords=["Anthropic", "OpenAI"],
    )
    # Missing keyword
    ok, reason = q.evaluate("Helios was created by someone.")
    assert ok is False
    assert "Missing required keyword" in reason

    # Forbidden keyword
    ok2, reason2 = q.evaluate("Helios was created by Arron Parejas at OpenAI.")
    assert ok2 is False
    assert "Contains forbidden keyword" in reason2


def test_benchmark_suite_size_and_coverage():
    assert len(BENCHMARK_SUITE) >= 50
    categories = {q.category for q in BENCHMARK_SUITE}
    assert "identity" in categories
    assert "greetings" in categories
    assert "logic" in categories
    assert "systems" in categories
    assert "math" in categories
    assert "filipino" in categories


def test_auto_improver_initialization():
    improver = AutoImprover(target_accuracy=0.95, max_rounds=2)
    assert improver.target_accuracy == 0.95
    assert improver.max_rounds == 2
