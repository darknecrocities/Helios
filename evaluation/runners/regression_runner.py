"""Regression runner comparing model checkpoints and flagging capability drops."""

from __future__ import annotations

from typing import Any

from evaluation.registry.manifests import BenchmarkManifest


class RegressionRunner:
    """Analyzes differences between checkpoint evaluation manifests."""

    @staticmethod
    def compare_manifests(
        current: BenchmarkManifest,
        baseline: BenchmarkManifest,
        regression_threshold_pct: float = 2.0,
    ) -> dict[str, Any]:
        """Compare current run against baseline manifest.

        Args:
            current: Newly evaluated manifest
            baseline: Previous reference/approved manifest
            regression_threshold_pct: Allowable performance decrease before flagging

        Returns:
            Dictionary with delta, status (PASS, REVIEW, FAIL), and regression notes
        """
        curr_score = current.results.score
        base_score = baseline.results.score
        delta = curr_score - base_score

        if delta < -regression_threshold_pct:
            status = "FAIL"
            message = (
                f"Severe regression detected in {current.benchmark.name}: "
                f"{curr_score:.1f}% vs baseline {base_score:.1f}% (drop of {abs(delta):.1f}%)"
            )
        elif delta < 0.0:
            status = "REVIEW"
            message = (
                f"Minor decrease in {current.benchmark.name}: "
                f"{curr_score:.1f}% vs baseline {base_score:.1f}% (drop of {abs(delta):.1f}%)"
            )
        else:
            status = "PASS"
            message = (
                f"Performance improved or sustained in {current.benchmark.name}: "
                f"{curr_score:.1f}% vs baseline {base_score:.1f}% (+{delta:.1f}%)"
            )

        return {
            "current_experiment": current.experiment_id,
            "baseline_experiment": baseline.experiment_id,
            "benchmark": current.benchmark.name,
            "current_score": curr_score,
            "baseline_score": base_score,
            "delta": delta,
            "status": status,
            "message": message,
        }
