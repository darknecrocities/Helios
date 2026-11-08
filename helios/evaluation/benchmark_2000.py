"""Helios 2,000-Question Multi-Subject Evaluation Benchmark Loader.

Loads the curated 2,000-question JSONL benchmark across 4 Core Pillars:
- STEM & Computing (600 items)
- Reasoning & Logic (500 items)
- Humanities & Social Sciences (450 items)
- General World Knowledge & Identity (450 items)

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
from collections.abc import Generator
from functools import lru_cache
from pathlib import Path

from helios.evaluation.schema import BenchmarkQuestion

DEFAULT_BENCHMARK_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "data"
    / "benchmarks"
    / "helios_2000_benchmark.jsonl"
)

PILLAR_STEM = "STEM & Computing"
PILLAR_REASONING = "Reasoning & Logic"
PILLAR_REASONING_LEGACY = "Multi-Step Reasoning"
PILLAR_HUMANITIES = "Humanities & Social Sciences"
PILLAR_KNOWLEDGE = "General World Knowledge & Identity"

PILLARS = [
    PILLAR_STEM,
    PILLAR_REASONING,
    PILLAR_HUMANITIES,
    PILLAR_KNOWLEDGE,
]


def resolve_benchmark_path(custom_path: Path | str | None = None) -> Path:
    """Resolve absolute path to the 2,000-question benchmark JSONL file."""
    if custom_path is not None:
        p = Path(custom_path).resolve()
        if p.exists():
            return p
    if DEFAULT_BENCHMARK_PATH.exists():
        return DEFAULT_BENCHMARK_PATH
    raise FileNotFoundError(f"Benchmark file not found at: {DEFAULT_BENCHMARK_PATH}")


def stream_2000_questions(
    benchmark_path: Path | str | None = None,
) -> Generator[BenchmarkQuestion, None, None]:
    """Stream benchmark items line-by-line without loading entire dataset into memory."""
    path = resolve_benchmark_path(benchmark_path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            yield BenchmarkQuestion(
                id=data["id"],
                pillar=data["pillar"],
                discipline=data["discipline"],
                subdiscipline=data["subdiscipline"],
                question=data["question"],
                target=data["target"],
                eval_mode=data.get("eval_mode", "contains"),
                tolerance=data.get("tolerance", 0.01),
                expected_pattern=data.get("expected_pattern", ""),
                reference_answer=data.get("reference_answer", data["target"]),
                explanation=data.get("explanation", ""),
                difficulty=data.get("difficulty", "medium"),
                language=data.get("language", "en"),
            )


def load_benchmark_questions(
    benchmark_path: Path | str | None = None,
) -> list[BenchmarkQuestion]:
    """Load all benchmark questions from JSONL file into a list."""
    return list(stream_2000_questions(benchmark_path))


def load_2000_benchmark(
    benchmark_path: Path | str | None = None,
) -> list[BenchmarkQuestion]:
    """Alias for load_benchmark_questions for API compatibility."""
    return load_benchmark_questions(benchmark_path)


@lru_cache(maxsize=1)
def get_all_2000_questions(
    benchmark_path: str | None = None,
) -> list[BenchmarkQuestion]:
    """Retrieve all 2,000 benchmark questions as a cached in-memory list."""
    return load_benchmark_questions(benchmark_path)


def get_2000_pillar_counts(
    questions: list[BenchmarkQuestion] | None = None,
) -> dict[str, int]:
    """Return question counts grouped by Core Pillar."""
    items = questions or get_all_2000_questions()
    counts: dict[str, int] = {}
    for q in items:
        counts[q.pillar] = counts.get(q.pillar, 0) + 1
    return counts


def get_2000_discipline_counts(
    questions: list[BenchmarkQuestion] | None = None,
) -> dict[str, int]:
    """Return question counts grouped by Discipline."""
    items = questions or get_all_2000_questions()
    counts: dict[str, int] = {}
    for q in items:
        counts[q.discipline] = counts.get(q.discipline, 0) + 1
    return counts


def get_questions_by_pillar(
    pillar: str, questions: list[BenchmarkQuestion] | None = None
) -> list[BenchmarkQuestion]:
    """Filter items belonging to a specific Core Pillar."""
    items = questions or get_all_2000_questions()
    return [q for q in items if q.pillar.lower() == pillar.lower()]


get_pillar_counts = get_2000_pillar_counts
get_discipline_counts = get_2000_discipline_counts
