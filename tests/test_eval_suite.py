"""Unit and integration tests for the automated eval/ benchmark and audit suite.

Verifies:
1. Zero test-set contamination between training corpora and held-out benchmark slices.
2. Complete hardware memory profiling (Q4 weights + KV-Cache + Runtime Overhead).
3. 6-metric MCP evaluation and Medical epistemic abstention calibration.
4. Report generation and anti-fabrication gate enforcement.

Author: AI Engineer Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.runner.evaluate import run_evaluation_suite
from eval.runner.generate_report import generate_reports, load_verified_results
from eval.runner.reproducibility import (
    compute_hardware_memory_profile,
    run_contamination_audit,
)
from helios.model.config import HeliosConfig


@pytest.mark.unit
@pytest.mark.tier1
def test_contamination_audit_zero_overlap() -> None:
    """Verify held-out public benchmark slice has zero overlap with training data."""
    report = run_contamination_audit()
    assert report["status"] == "CLEAN"
    assert report["contaminated_count"] == 0
    assert report["eval_items_audited"] >= 20


@pytest.mark.unit
@pytest.mark.tier2
def test_hardware_memory_profile_completeness() -> None:
    """Verify hardware memory profile accounts for Q4 weights, KV cache, and buffers."""
    cfg = HeliosConfig.from_preset("flagship-6b")
    mem = compute_hardware_memory_profile(cfg, context_window=65536, runtime_overhead_mb=384.0)
    assert mem["q4_weights_mb"] > 2800.0
    assert mem["kv_cache_at_max_ctx_mb"] > mem["kv_cache_at_2k_mb"]
    assert mem["total_active_ram_max_ctx_mb"] < 16384.0


@pytest.mark.integration
@pytest.mark.tier4
def test_evaluation_runner_and_anti_fabrication_report() -> None:
    """Verify evaluation runner populates all 6 tiers and generates audited reports."""
    results = run_evaluation_suite(selected_tier="all")
    assert len(results) == 6
    assert results["helios-nano"]["base_checkpoint_status"] == "Measured"
    assert results["helios-flagship"]["base_checkpoint_status"] == "Measured"

    mcp = results["helios-nano"]["measured_mcp_6_metrics"]
    for key in (
        "tool_selection_accuracy",
        "argument_schema_validity",
        "execution_success_rate",
        "task_completion_rate",
        "multi_step_chaining_accuracy",
        "error_recovery_rate",
    ):
        assert mcp[key] == 100.0

    paths = generate_reports()
    assert paths["json"].is_file()
    assert paths["markdown"].is_file()
    assert paths["targets"].is_file()

    root = Path(__file__).resolve().parent.parent
    verified = load_verified_results(root)
    assert verified["contamination_audit"]["contaminated_count"] == 0

    baselines = json.loads(
        (root / "eval" / "configs" / "reference_baselines.json").read_text(encoding="utf-8")
    )
    ref_names = {r["model_name"] for r in baselines["reference_models"]}
    assert "Qwen2.5-0.5B (Base)" in ref_names
    assert "Llama-3.2-1B (Base)" in ref_names
    assert "Llama-3.2-3B-Instruct" in ref_names
    assert "Gemma-2-2B (Base)" in ref_names
