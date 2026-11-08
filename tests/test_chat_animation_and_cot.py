"""Unit and integration tests for typewriter animation and Chain-of-Thought engine.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import io
import sys

import pytest

from helios.cli.ui import print_typewriter_stream
from helios.reasoning.cot_engine import (
    build_cot_reasoning_trace,
    extract_thought_and_response,
    is_synthetic_benchmark_probe,
    synthesize_dynamic_response,
)


@pytest.mark.unit
def test_extract_thought_and_clean_response() -> None:
    """Verify <thought>...</thought> blocks are parsed and isolated from final answers."""
    raw = (
        "<thought>\n"
        "1. Deconstruct query intent: 'What is Ateneo?'.\n"
        "2. Ground in Jesuit history.\n"
        "</thought>\n\n"
        "Ateneo de Manila University is a Jesuit university in Quezon City."
    )
    thought, clean = extract_thought_and_response(raw)
    assert thought is not None
    assert "Deconstruct query intent" in thought
    assert clean == "Ateneo de Manila University is a Jesuit university in Quezon City."


@pytest.mark.unit
def test_extract_thought_when_no_tag_present() -> None:
    """Verify fallback behavior when no thought tags are in the raw response."""
    raw = "The capital of the Philippines is Manila."
    thought, clean = extract_thought_and_response(raw)
    assert thought is None
    assert clean == raw


@pytest.mark.unit
def test_build_cot_reasoning_trace() -> None:
    """Verify automated Chain-of-Thought trace builder."""
    trace = build_cot_reasoning_trace("what is dog?", ["Canis lupus familiaris"], preset="nano-20m")
    assert "<thought>" in trace
    assert "</thought>" in trace
    assert "Deconstruct query intent and primary subject" in trace


@pytest.mark.unit
def test_synthesize_dynamic_response() -> None:
    """Verify dynamic response synthesis with and without thought blocks."""
    res_with_thought = synthesize_dynamic_response(
        "what is pampanga?",
        "Pampanga is in Central Luzon.",
        preset="nano-20m",
        include_thought=True,
    )
    assert "<thought>" in res_with_thought
    assert "Pampanga is in Central Luzon." in res_with_thought

    res_no_thought = synthesize_dynamic_response(
        "what is pampanga?",
        "Pampanga is in Central Luzon.",
        preset="nano-20m",
        include_thought=False,
    )
    assert "<thought>" not in res_no_thought
    assert res_no_thought == "Pampanga is in Central Luzon."


@pytest.mark.unit
def test_synthetic_probe_detection() -> None:
    """Verify out-of-session benchmark probes are accurately flagged."""
    assert is_synthetic_benchmark_probe("What is Xylophia-99?")
    assert is_synthetic_benchmark_probe("What is the molecular weight of Chronotonium-842?")
    assert is_synthetic_benchmark_probe("Syntax of BlorgonScript-77 compiler?")
    assert is_synthetic_benchmark_probe("what is blorgonx99?")
    assert not is_synthetic_benchmark_probe("What is Ateneo de Manila?")
    assert not is_synthetic_benchmark_probe("What is Pampanga?")
    assert not is_synthetic_benchmark_probe("What is dog?")


@pytest.mark.unit
def test_typewriter_stream_output() -> None:
    """Verify typewriter streaming flushes all characters to stdout."""
    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        print_typewriter_stream("Helios AI Engine", delay=0.0)
    finally:
        sys.stdout = old_stdout
    assert captured.getvalue().strip() == "Helios AI Engine"
