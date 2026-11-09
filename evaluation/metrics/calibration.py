"""Calibration error (ECE) and uncertainty metrics for Epistemic Honesty."""

from __future__ import annotations

import numpy as np


def compute_calibration_error(
    confidences: list[float],
    accuracies: list[bool],
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: Predicted confidence probability for each sample (0.0 to 1.0)
        accuracies: Binary boolean accuracy for each sample
        num_bins: Number of confidence partition bins

    Returns:
        ECE value between 0.0 and 1.0 (lower is better)
    """
    if not confidences or len(confidences) != len(accuracies):
        return 0.0

    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies, dtype=float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_mask = (conf_arr > bin_boundaries[i]) & (conf_arr <= bin_boundaries[i + 1])
        bin_count = np.sum(bin_mask)
        if bin_count > 0:
            bin_acc = np.mean(acc_arr[bin_mask])
            bin_conf = np.mean(conf_arr[bin_mask])
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)
