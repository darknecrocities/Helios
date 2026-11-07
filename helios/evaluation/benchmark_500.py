"""Helios 500-Question Unique Non-Biased Evaluation Benchmark.

Aggregates 500 rigorously calibrated, non-biased questions across 11 comprehensive
domains (including Medical & Clinical Sciences, Physical Sciences, Mathematics,
Syllogistic Deduction, Causal Reasoning, Computer Science & Tech Entities,
Critical Thinking, World Geography, Bilingual English/Tagalog & Model Identity,
Spatial/Temporal Logic, and Epistemic Fallback Verification).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from helios.evaluation.benchmark_300_part1 import PART1_QUESTIONS
from helios.evaluation.benchmark_300_part2 import PART2_QUESTIONS
from helios.evaluation.benchmark_500_part3 import PART3_QUESTIONS
from helios.evaluation.benchmark_500_part4 import PART4_QUESTIONS
from helios.evaluation.schema import BenchmarkQuestion

ALL_500_QUESTIONS: list[BenchmarkQuestion] = (
    PART1_QUESTIONS + PART2_QUESTIONS + PART3_QUESTIONS + PART4_QUESTIONS
)

DOMAINS_500: list[str] = [
    "Medical, Clinical & Life Sciences",
    "General & Physical Science",
    "Math & Quantitative Reasoning",
    "Contextual & Syllogistic Deduction",
    "Causal & Counterfactual Reasoning",
    "Computer Science & Systems",
    "Critical Thinking & Formal Fallacies",
    "World Knowledge & Geography",
    "Bilingual English/Tagalog Comprehension",
    "Spatial & Temporal Logic",
    "Ethical, Neutral & Unbiased Reasoning",
]


def get_all_500_questions() -> list[BenchmarkQuestion]:
    """Retrieve the full set of 500 unique benchmark questions."""
    return ALL_500_QUESTIONS


def get_500_domain_counts() -> dict[str, int]:
    """Count number of questions per domain in the 500-question suite."""
    counts: dict[str, int] = {}
    for q in ALL_500_QUESTIONS:
        counts[q.domain] = counts.get(q.domain, 0) + 1
    return counts
