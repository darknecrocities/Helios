"""Helios reasoning and Chain-of-Thought (CoT) orchestration module."""

from helios.reasoning.cot_engine import (
    build_cot_reasoning_trace,
    extract_thought_and_response,
    is_synthetic_benchmark_probe,
    synthesize_dynamic_response,
)
from helios.reasoning.intent import (
    QueryIntent,
    classify_query_intent,
    clean_meta_prompt_answer,
    is_candidate_intent_compatible,
    score_qa_alignment,
    synthesize_first_principles_definition,
)

__all__ = [
    "QueryIntent",
    "build_cot_reasoning_trace",
    "classify_query_intent",
    "clean_meta_prompt_answer",
    "extract_thought_and_response",
    "is_candidate_intent_compatible",
    "is_synthetic_benchmark_probe",
    "score_qa_alignment",
    "synthesize_dynamic_response",
    "synthesize_first_principles_definition",
]
