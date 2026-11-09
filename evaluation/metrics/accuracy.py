"""Accuracy and F1 metrics for classification and multiple-choice benchmarks."""

from __future__ import annotations


def compute_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Compute standard top-1 classification accuracy percentage (0.0 to 100.0)."""
    if not predictions or len(predictions) != len(targets):
        return 0.0

    correct = 0
    for pred, tgt in zip(predictions, targets, strict=False):
        clean_pred = pred.strip().upper()
        clean_tgt = tgt.strip().upper()
        if clean_pred == clean_tgt or clean_pred.startswith(clean_tgt):
            correct += 1

    return (correct / len(predictions)) * 100.0
