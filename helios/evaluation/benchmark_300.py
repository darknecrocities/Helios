"""Helios 300-Question Non-Biased Evaluation Benchmark.

Aggregates 300 rigorously calibrated, non-biased questions across 10 balanced domains
(30 questions per domain) to evaluate deduction, causal inference, mathematics,
world knowledge, bilingual comprehension, and ethical reasoning.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from helios.evaluation.benchmark_300_part1 import PART1_QUESTIONS
from helios.evaluation.benchmark_300_part2 import PART2_QUESTIONS
from helios.evaluation.schema import BenchmarkQuestion

ALL_300_QUESTIONS: list[BenchmarkQuestion] = PART1_QUESTIONS + PART2_QUESTIONS

DOMAINS: list[str] = [
    "Math & Quantitative Reasoning",
    "Contextual & Syllogistic Deduction",
    "Causal & Counterfactual Reasoning",
    "General & Physical Science",
    "Computer Science & Systems",
    "Critical Thinking & Formal Fallacies",
    "World Knowledge & Geography",
    "Bilingual English/Tagalog Comprehension",
    "Spatial & Temporal Logic",
    "Ethical, Neutral & Unbiased Reasoning",
]


def get_all_questions() -> list[BenchmarkQuestion]:
    """Retrieve the full set of 300 benchmark questions."""
    return ALL_300_QUESTIONS


def get_questions_by_domain(domain: str) -> list[BenchmarkQuestion]:
    """Retrieve questions filtered by domain name."""
    return [q for q in ALL_300_QUESTIONS if q.domain.lower() == domain.lower()]


def get_domain_counts() -> dict[str, int]:
    """Count number of questions per domain."""
    counts: dict[str, int] = {}
    for q in ALL_300_QUESTIONS:
        counts[q.domain] = counts.get(q.domain, 0) + 1
    return counts
