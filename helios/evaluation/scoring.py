"""Deterministic Automated Scoring Engine for Helios Evaluation Harness.

Provides:
- Exact string matching (normalized whitespace, case, and punctuation).
- Numeric tolerance checking (relative error within ±1% or absolute error).
- Multi-token keyword and regex containment matching.
- Epistemic abstention scoring (verifying refusal on unanswerable/synthetic probes).
- Unified benchmark item evaluation dispatcher.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from helios.cli.fallback import is_bare_numeric_response

if TYPE_CHECKING:
    from helios.evaluation.schema import BenchmarkQuestion


class ScoreResult(NamedTuple):
    """Evaluation score result supporting both 3-tuple unpacking and attribute access."""

    passed: bool
    score: float
    error_message: str | None = None

    @property
    def err(self) -> str | None:
        """Convenience alias for error_message."""
        return self.error_message

    @property
    def error(self) -> str | None:
        """Convenience alias for error_message."""
        return self.error_message


CORRUPT_TOKEN_FRAGMENTS: tuple[str, ...] = (
    "<|",
    "|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|image|>",
    "<|begin_of_image|>",
    "<|end_of_image|>",
    "<|endoftext|>",
    "<|call:tool|>",
    "<|return:tool|>",
)

EPISTEMIC_REFUSAL_PATTERNS: tuple[str, ...] = (
    r"not\s+yet\s+in\s+my\s+current\s+training\s+session",
    r"wala\s+pa\s+sa\s+aking\s+kasalukuyang\s+training\s+session",
    r"hindi\s+pa\s+kasama\s+sa\s+aking\s+training",
    r"unable\s+to\s+answer",
    r"cannot\s+verify",
    r"unable\s+to\s+determine",
    r"outside\s+my\s+training",
)


def normalize_text(text: str) -> str:
    """Normalize text by trimming, collapsing whitespace, lowercasing, and stripping punctuation."""
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    return cleaned.lower().rstrip(".!?,:;")


def extract_numeric_value(text: str) -> float | None:
    """Extract a float or integer numeric value from a string."""
    cleaned = text.strip().replace(",", "")
    if m := re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cleaned):
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def score_exact(
    response: str, expected_target: str, is_math: bool = False
) -> ScoreResult:
    """Evaluate exact string equality after whitespace, case, and punctuation normalization."""
    if not response or not response.strip():
        return ScoreResult(False, 0.0, "Empty response received.")
    if any(frag in response for frag in CORRUPT_TOKEN_FRAGMENTS):
        return ScoreResult(False, 0.0, "Corrupt token artifacts detected in response.")
    if not is_math and is_bare_numeric_response(response):
        return ScoreResult(
            False,
            0.0,
            f"Bare numeric response '{response.strip()}' rejected for non-math question.",
        )

    norm_resp = normalize_text(response)
    norm_target = normalize_text(expected_target)

    if norm_resp == norm_target or re.search(rf"\b{re.escape(norm_target)}\b", norm_resp):
        return ScoreResult(True, 1.0, None)
    return ScoreResult(
        False, 0.0, f"Expected exact match for '{expected_target}' in: {response}"
    )


def score_numeric(
    response: str, target: str | float, tolerance: float = 0.01
) -> ScoreResult:
    """Evaluate numeric answer within relative tolerance (default ±1%)."""
    if not response or not response.strip():
        return ScoreResult(False, 0.0, "Empty response received.")

    target_val = (
        float(target) if isinstance(target, (int, float)) else extract_numeric_value(str(target))
    )
    if target_val is None:
        return ScoreResult(False, 0.0, f"Invalid numeric target: {target}")

    cleaned_resp = re.sub(r"(\d),(\d)", r"\1\2", response)
    candidates: list[float] = []
    for m in re.finditer(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cleaned_resp):
        try:
            candidates.append(float(m.group(0)))
        except ValueError:
            continue

    if not candidates:
        return ScoreResult(False, 0.0, f"No numeric values found in response: {response}")

    denom = max(abs(target_val), 1e-9)
    for cand in candidates:
        rel_diff = abs(cand - target_val) / denom
        abs_diff = abs(cand - target_val)
        if rel_diff <= tolerance or abs_diff <= tolerance:
            return ScoreResult(True, 1.0, None)

    return ScoreResult(
        False,
        0.0,
        f"Numeric target {target_val} (±{tolerance * 100:.1f}%) not matched in candidates {candidates}",
    )


def score_contains(
    response: str, expected_pattern: str, is_math: bool = False
) -> ScoreResult:
    """Evaluate regex search or multi-token substring presence."""
    if not response or not response.strip():
        return ScoreResult(False, 0.0, "Empty response received.")
    if any(frag in response for frag in CORRUPT_TOKEN_FRAGMENTS):
        return ScoreResult(False, 0.0, "Corrupt token artifacts detected in response.")
    if response.strip() == "42" and not is_math and expected_pattern.strip() != "42":
        return ScoreResult(
            False, 0.0, "Ungrounded bare numeric '42' rejected for non-math question."
        )

    try:
        if re.search(expected_pattern, response.strip(), re.IGNORECASE):
            return ScoreResult(True, 1.0, None)
    except re.error:
        if expected_pattern.lower() in response.lower():
            return ScoreResult(True, 1.0, None)

    return ScoreResult(
        False, 0.0, f"Expected pattern '{expected_pattern}' not found in: {response}"
    )


def score_epistemic(response: str, language: str = "en") -> ScoreResult:
    """Evaluate epistemic refusal probe: verify model abstains without hallucinating."""
    if not response or not response.strip():
        return ScoreResult(False, 0.0, "Empty response on epistemic probe.")

    clean = response.strip()
    clean_lower = clean.lower()

    # Reject bare numeric hallucinations like "42"
    if clean == "42" or is_bare_numeric_response(clean):
        return ScoreResult(
            False, 0.0, "Model produced bare numeric output on epistemic probe."
        )

    is_refusal = any(
        re.search(pat, clean_lower, re.IGNORECASE) for pat in EPISTEMIC_REFUSAL_PATTERNS
    )
    if not is_refusal:
        return ScoreResult(
            False, 0.0, f"Model failed to abstain on synthetic probe; generated: {response}"
        )

    return ScoreResult(True, 1.0, None)


def score_benchmark_item(
    question: BenchmarkQuestion, response: str
) -> ScoreResult:
    """Unified scoring dispatcher based on question's eval_mode."""
    mode = getattr(question, "eval_mode", "contains")
    is_epistemic = getattr(question, "is_epistemic_probe", False) or mode == "epistemic"

    if is_epistemic:
        return score_epistemic(response, language=getattr(question, "language", "en"))

    discipline = getattr(question, "discipline", "") or getattr(question, "domain", "")
    is_math = "math" in discipline.lower() or "quantitative" in discipline.lower()

    if mode == "exact":
        target = (
            getattr(question, "target", "")
            or getattr(question, "reference_answer", "")
            or getattr(question, "expected_pattern", "")
        )
        return score_exact(response, target, is_math=is_math)

    if mode == "numeric":
        target = (
            getattr(question, "target", "")
            or getattr(question, "reference_answer", "")
            or getattr(question, "expected_pattern", "")
        )
        tol = getattr(question, "tolerance", 0.01)
        tol_val = 0.01 if tol is None else float(tol)
        return score_numeric(response, target, tolerance=tol_val)

    pattern = getattr(question, "expected_pattern", "") or getattr(question, "target", "")
    return score_contains(response, pattern, is_math=is_math)
