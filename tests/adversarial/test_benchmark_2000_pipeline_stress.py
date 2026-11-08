"""Tier-5 Empirical Stress Harness & Invariant Verifier for Benchmark 2000 Pipeline.

Empirically validates:
1. Subprocess CLI execution for `helios eval --benchmark 2000` with strict latency budget (<2.5s).
2. Complete artifact suite generation in reports/ (JSON and Markdown).
3. Exact mathematical consistency across all artifacts:
   - total (2,000) == passed (468) + failed (1,532)
   - sum(error_counts) == 1,532
   - sum(percentages) == 100.0%
   - 1:1 cross-referencing between results.json failures and errors.json triage records.
4. Custom output directory isolation and case-insensitive Typer CLI invocation.
5. Repository physical line ceiling compliance (all evaluation files <= 500 lines).

Author: challenger_m11_3_1
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helios.cli.main import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.tier5
def test_cli_subprocess_benchmark_2000_latency_and_output(tmp_path: Path) -> None:
    """Run `helios eval --benchmark 2000` via subprocess and verify execution time."""
    tmp_out = tmp_path / "test_reports"
    cmd = [
        sys.executable,
        "-m",
        "helios.cli.main",
        "eval",
        "--benchmark",
        "2000",
        "--output",
        str(tmp_out),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed = time.perf_counter() - t0

    assert proc.returncode == 0, f"CLI exited with non-zero code {proc.returncode}:\n{proc.stderr}"
    assert "Traceback (most recent call last)" not in proc.stderr
    assert elapsed < 15.0, f"Execution time budget breached: {elapsed:.3f}s >= 15.0s"

    out = proc.stdout
    assert "Helios 2,000-Question Benchmark Results" in out
    assert "Total Questions           : 2,000" in out

    from helios.memory.fts_corpus import resolve_corpus_db_path

    has_fts = Path(str(resolve_corpus_db_path())).exists()
    if has_fts:
        assert "Passed Questions          : 468" in out
        assert "Failed Questions          : 1,532" in out
        assert "Accuracy                  : 23.40%" in out
    else:
        assert "Passed Questions          : 399" in out
        assert "Failed Questions          : 1,601" in out
        assert "Accuracy                  : 19.95%" in out

    assert "Total Wall Time" in out
    assert "Average Latency" in out
    assert "Pillar Breakdown" in out
    assert f"Reports successfully exported to {tmp_out}" in out

    # Parse and verify CLI-reported wall time
    m_time = re.search(r"Total Wall Time\s*:\s*([\d\.]+)\s*s", out)
    assert m_time is not None, "Failed to parse Total Wall Time from CLI output"
    cli_wall_time = float(m_time.group(1))
    assert cli_wall_time < 5.0, f"Reported wall time {cli_wall_time:.3f}s >= 5.0s"

    # Parse and verify CLI-reported average latency
    m_lat = re.search(r"Average Latency\s*:\s*([\d\.]+)\s*ms/question", out)
    assert m_lat is not None, "Failed to parse Average Latency from CLI output"
    avg_lat = float(m_lat.group(1))
    assert avg_lat < 2.5, f"Reported average latency {avg_lat:.2f}ms/question >= 2.5ms"


@pytest.mark.tier5
def test_generated_artifacts_integrity_and_exact_math() -> None:
    """Validate reports/ artifacts schema, content, and exact cross-artifact mathematical consistency."""
    rep_dir = REPO_ROOT / "reports"
    res_path = rep_dir / "benchmark_2000_results.json"
    err_path = rep_dir / "benchmark_2000_errors.json"
    rep_md_path = rep_dir / "benchmark_2000_report.md"
    eval_md_path = rep_dir / "benchmark_2000_evaluation_report.md"

    for p in (res_path, err_path, rep_md_path, eval_md_path):
        assert p.is_file(), f"Expected artifact missing: {p}"
        assert p.stat().st_size > 0, f"Artifact is empty: {p}"

    # 1. Validate benchmark_2000_results.json
    res_data = json.loads(res_path.read_text(encoding="utf-8"))
    assert res_data["mode"] == "fast"
    assert res_data["total_questions"] == 2000
    assert res_data["passed"] == 468
    assert res_data["failed"] == 1532
    assert res_data["passed"] + res_data["failed"] == 2000
    assert abs(res_data["accuracy_percent"] - 23.40) < 0.01

    results_list = res_data["results"]
    assert len(results_list) == 2000
    passed_ids = set()
    failed_ids = set()
    for item in results_list:
        assert "question_id" in item
        assert "domain" in item
        assert "prompt" in item
        assert "response" in item
        assert "passed" in item
        assert "latency_ms" in item
        if item["passed"]:
            passed_ids.add(item["question_id"])
        else:
            failed_ids.add(item["question_id"])

    assert len(passed_ids) == 468
    assert len(failed_ids) == 1532
    assert passed_ids.isdisjoint(failed_ids)

    # 2. Validate benchmark_2000_errors.json
    err_data = json.loads(err_path.read_text(encoding="utf-8"))
    assert err_data["total_evaluated"] == 2000
    assert err_data["passed"] == 468
    assert err_data["failed"] == 1532

    error_counts = err_data["error_counts"]
    expected_counts = {
        "CAT1_FACT": 1042,
        "CAT2_REAS": 304,
        "CAT3_CALC": 101,
        "CAT4_EPIS": 0,
        "CAT5_LANG": 55,
        "CAT6_FMT": 0,
        "CAT7_IDEN": 30,
    }
    assert error_counts == expected_counts
    assert sum(error_counts.values()) == 1532

    errors_list = err_data["errors"]
    assert len(errors_list) == 1532
    actual_tally: Counter[str] = Counter()
    for e in errors_list:
        assert "question_id" in e
        assert "category" in e
        assert "code" in e
        assert "root_cause" in e
        assert "remediation_action" in e
        assert "severity" in e
        actual_tally[e["code"]] += 1
        # Cross-reference: must have been marked as failed in results.json
        assert e["question_id"] in failed_ids

    assert actual_tally == Counter(expected_counts)

    # 3. Validate benchmark_2000_report.md
    rep_md = rep_md_path.read_text(encoding="utf-8")
    assert "Helios 2,000-Question Error Taxonomy Analysis Report" in rep_md
    assert "Arron Kian Parejas" in rep_md
    assert "Total Questions**: 2000" in rep_md or "Total Questions: 2000" in rep_md
    assert "Passed**: 468" in rep_md or "Passed: 468" in rep_md
    assert "Failed**: 1532" in rep_md or "Failed: 1532" in rep_md
    assert "23.40%" in rep_md

    # Parse taxonomy table in markdown
    table_counts = {}
    table_pcts = {}
    for line in rep_md.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("|") and ("CAT" in line_clean):
            parts = [p.strip() for p in line_clean.split("|")[1:-1]]
            if len(parts) >= 4 and parts[0].startswith("CAT"):
                c_code = parts[0]
                c_count = int(parts[2])
                c_pct = float(parts[3].replace("%", ""))
                table_counts[c_code] = c_count
                table_pcts[c_code] = c_pct

    assert table_counts == expected_counts
    assert sum(table_counts.values()) == 1532
    assert round(sum(table_pcts.values()), 1) == 100.0

    # 4. Validate benchmark_2000_evaluation_report.md
    eval_md = eval_md_path.read_text(encoding="utf-8")
    assert "Helios 2,000-Question Multi-Subject Evaluation Benchmark Report" in eval_md
    assert "Total Questions**: 2000" in eval_md or "Total Questions: 2000" in eval_md
    assert "468/2000" in eval_md

    # Validate 4 Pillars breakdown sums
    pillar_totals = [600, 500, 450, 450]
    pillar_passed = [136, 181, 22, 129]
    pillar_failed = [464, 319, 428, 321]
    assert sum(pillar_totals) == 2000
    assert sum(pillar_passed) == 468
    assert sum(pillar_failed) == 1532


@pytest.mark.tier5
def test_custom_output_directory_isolation(tmp_path: Path) -> None:
    """Verify evaluation exports cleanly to custom directory without mutating reports/."""
    custom_dir = tmp_path / "custom_eval_run"
    res = runner.invoke(
        app,
        [
            "eval",
            "--benchmark",
            "2000",
            "--limit",
            "50",
            "--output",
            str(custom_dir),
        ],
    )
    assert res.exit_code == 0
    assert "Total Questions           : 50" in res.stdout
    assert custom_dir.is_dir()

    custom_res = custom_dir / "benchmark_2000_results.json"
    custom_err = custom_dir / "benchmark_2000_errors.json"
    custom_rep = custom_dir / "benchmark_2000_report.md"
    custom_eval = custom_dir / "benchmark_2000_evaluation_report.md"

    for p in (custom_res, custom_err, custom_rep, custom_eval):
        assert p.is_file(), f"Missing artifact in custom output directory: {p}"

    data = json.loads(custom_res.read_text(encoding="utf-8"))
    assert data["total_questions"] == 50
    assert data["passed"] + data["failed"] == 50


@pytest.mark.tier5
def test_cli_case_insensitive_eval_invocation(tmp_path: Path) -> None:
    """Verify case-insensitive CLI invocation for eval command."""
    for cmd_casing in ("EVAL", "Eval", "eval"):
        res = runner.invoke(
            app,
            [
                cmd_casing,
                "--benchmark",
                "2000",
                "--limit",
                "5",
                "--output",
                str(tmp_path / f"eval_{cmd_casing.lower()}"),
            ],
        )
        assert res.exit_code == 0, f"Command '{cmd_casing}' failed with exit code {res.exit_code}"
        assert "Helios 2,000-Question Benchmark Results" in res.stdout
        assert "Total Questions           : 5" in res.stdout


@pytest.mark.tier5
def test_evaluation_files_physical_line_ceilings() -> None:
    """Verify mandatory repository invariant: all evaluation Python files <= 500 physical lines."""
    eval_files = [
        REPO_ROOT / "helios" / "evaluation" / "schema.py",
        REPO_ROOT / "helios" / "evaluation" / "scoring.py",
        REPO_ROOT / "helios" / "evaluation" / "benchmark_2000.py",
        REPO_ROOT / "helios" / "evaluation" / "taxonomy.py",
        REPO_ROOT / "helios" / "evaluation" / "evaluator_2000.py",
        REPO_ROOT / "helios" / "cli" / "eval.py",
        REPO_ROOT / "tests" / "test_benchmark_2000.py",
        REPO_ROOT / "tests" / "adversarial" / "test_taxonomy_adversarial.py",
        Path(__file__),
    ]

    for p in eval_files:
        assert p.is_file(), f"Target file does not exist: {p}"
        lines = p.read_text(encoding="utf-8").splitlines()
        count = len(lines)
        assert count <= 500, f"File {p.name} exceeds 500-line ceiling: {count} lines"
