"""Pass@k metric for code synthesis and sandbox execution."""

from __future__ import annotations

import math


def compute_pass_at_k(n: int, c: int, k: int = 1) -> float:
    """Compute unbiased pass@k metric (Chen et al., HumanEval).

    Args:
        n: Total number of samples generated per problem
        c: Number of correct samples passing all unit tests
        k: k in pass@k

    Returns:
        Unbiased pass@k probability (0.0 to 100.0)
    """
    if n - c < k:
        return 100.0
    val = 1.0 - math.comb(n - c, k) / math.comb(n, k)
    return val * 100.0
