"""Unit and integration tests for the HELIOS Benchmarking Platform."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from evaluation.cli import app
from evaluation.registry.checkpoints import CheckpointVerifier
from evaluation.registry.manifests import (
    BenchmarkManifest,
    BenchmarkResults,
    BenchmarkSpec,
    EnvironmentSpec,
    InferenceSpec,
    ModelRecord,
)
from evaluation.reporting.comparison import (
    format_delta,
    generate_comparison_markdown,
)
from evaluation.reporting.html import generate_html_report
from evaluation.reporting.markdown import generate_markdown_report
from evaluation.tasks.academic.mmlu import MMLUTask
from evaluation.tasks.coding.humaneval import HumanEvalTask
from evaluation.tasks.math.gsm8k import GSM8KTask
from evaluation.tasks.mcp.bfcl_suite import BFCLSuiteTask
from evaluation.tasks.medical.medmcqa import MedMCQATask
from evaluation.tasks.memory.hma_suite import HMASuiteTask
from evaluation.tasks.science.sciq import SciQTask


def test_checkpoint_verifier_inspect_yaml():
    """Verify CheckpointVerifier correctly parses model configuration YAML."""
    verifier = CheckpointVerifier()
    info = verifier.inspect("configs/models/nano.yaml")

    assert info.model_name == "Helios Nano"
    assert info.tier.upper() == "NANO"
    assert info.num_layers == 12
    assert info.hidden_dim == 768
    assert info.num_attention_heads == 12
    assert info.num_key_value_heads == 4
    assert len(info.weight_sha256) == 64
    assert info.format == "yaml_spec"


def test_benchmark_manifest_serialization():
    """Verify BenchmarkManifest Pydantic model serialization and roundtrip."""
    manifest = BenchmarkManifest(
        experiment_id="exp-test-001",
        model=ModelRecord(
            name="Helios Test",
            preset="nano-20m",
            parameters=20_000_000,
            checkpoint_sha256="abcdef1234567890",
            tokenizer_sha256="tok123",
            architecture="HeliosForCausalLM",
        ),
        benchmark=BenchmarkSpec(
            name="MMLU",
            samples_evaluated=10,
            few_shot=0,
        ),
        inference=InferenceSpec(
            temperature=0.0,
            seed=42,
            max_new_tokens=64,
            precision="float32",
        ),
        environment=EnvironmentSpec(
            device="cpu",
            gpu_name=None,
            python_version="3.12.0",
            torch_version="2.5.0",
            platform="Darwin",
        ),
        results=BenchmarkResults(
            score=72.5,
            metric_name="accuracy",
            status="completed",
            error_count=0,
            raw_samples=[],
        ),
    )

    data_json = manifest.model_dump_json()
    assert "exp-test-001" in data_json
    assert "72.5" in data_json

    restored = BenchmarkManifest.model_validate_json(data_json)
    assert restored.experiment_id == "exp-test-001"
    assert restored.results.score == 72.5


def test_task_loading():
    """Verify each benchmark task loads canonical sample items."""
    tasks = [
        MMLUTask(),
        GSM8KTask(),
        SciQTask(),
        MedMCQATask(),
        HumanEvalTask(),
        HMASuiteTask(),
        BFCLSuiteTask(),
    ]

    for task in tasks:
        samples = task.load_samples(limit=2)
        assert len(samples) > 0
        assert samples[0].sample_id is not None
        assert len(samples[0].prompt) > 0


def test_reporting_generators(tmp_path: Path):
    """Verify Markdown and HTML evaluation reports render cleanly."""
    mock_manifest = {
        "model_name": "Helios Nano",
        "model_tier": "nano",
        "run_id": "run-test-999",
        "timestamp": "2026-11-09T12:00:00Z",
        "aggregate_score": 0.684,
        "environment": {"device": "cpu", "torch_version": "2.5.0"},
        "results": {
            "MMLU": {
                "completed": True,
                "metrics": {"accuracy": 0.428, "expected_calibration_error": 0.045},
                "latency": {"tokens_per_second": 240.5},
                "memory": {"peak_allocated_mb": 61.3},
            },
            "GSM8K": {
                "completed": True,
                "metrics": {"accuracy": 0.485},
                "latency": {"tokens_per_second": 235.0},
                "memory": {"peak_allocated_mb": 62.1},
            },
        },
    }

    md_report = generate_markdown_report(mock_manifest)
    assert "Helios Nano" in md_report
    assert "MMLU" in md_report

    html_report = generate_html_report(mock_manifest)
    assert "<!DOCTYPE html>" in html_report
    assert "Helios Nano" in html_report
    assert "68.4%" in html_report


def test_comparison_generator():
    """Verify cross-run and baseline comparison markdown generation."""
    cand = {
        "model_name": "Helios Pro",
        "model_tier": "pro",
        "run_id": "cand-001",
        "timestamp": "2026-11-09",
        "aggregate_score": 0.785,
        "results": {"MMLU": {"metrics": {"accuracy": 0.762}}},
    }
    base = {
        "model_name": "Helios Core",
        "model_tier": "core",
        "run_id": "base-001",
        "timestamp": "2026-11-08",
        "aggregate_score": 0.690,
        "results": {"MMLU": {"metrics": {"accuracy": 0.665}}},
    }

    comp_md = generate_comparison_markdown(cand, base)
    assert "Helios Pro vs Helios Core" in comp_md
    assert "+9.7%" in comp_md

    assert format_delta(0.75, 0.70) == "+5.0%"
    assert format_delta(0.60, 0.65) == "-5.0%"


def test_cli_inspect_command():
    """Verify Typer CLI inspect command executes cleanly."""
    runner = CliRunner()
    result = runner.invoke(app, ["inspect", "configs/models/nano.yaml"])
    assert result.exit_code == 0
    assert "Helios Nano" in result.output
    assert "NANO" in result.output
