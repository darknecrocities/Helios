"""Exact match metric with robust answer extraction for GSM8K and math tasks."""

from __future__ import annotations

import re


def extract_gsm8k_answer(text: str) -> str:
    """Extract numeric answer from model generation using canonical GSM8K extraction."""
    # Look for '#### <number>' pattern
    if match := re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text):
        return match.group(1).replace(",", "").strip()

    # Look for 'The answer is <number>'
    if match := re.search(
        r"(?:the answer is|the result is|equals|=)\s*[:\s]*(-?[\d,]+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    ):
        return match.group(1).replace(",", "").strip()

    # Fallback to the last standalone number in the response
    numbers = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return text.strip()


def compute_exact_match(predictions: list[str], targets: list[str]) -> float:
    """Compute exact match percentage across normalized answers."""
    if not predictions or len(predictions) != len(targets):
        return 0.0

    matches = 0
    for pred, tgt in zip(predictions, targets, strict=False):
        norm_pred = extract_gsm8k_answer(pred)
        norm_tgt = extract_gsm8k_answer(tgt)

        try:
            if abs(float(norm_pred) - float(norm_tgt)) < 1e-4:
                matches += 1
                continue
        except ValueError:
            pass

        if norm_pred.lower() == norm_tgt.lower():
            matches += 1

    return (matches / len(predictions)) * 100.0
