"""Unit, Integration, and Latency Tests for Helios 2,000-Question Multi-Subject Benchmark.

Verifies:
1. Dataset completeness, uniqueness, 4-pillar distribution, creator attribution, and epistemic probes.
2. Automated scoring engine across all evaluation modes (exact, numeric, contains, epistemic).
3. 7-Type error taxonomy classification and priority-ordered triage.
4. Fast HMA Memory Mode execution latency (<2.5s budget for all 2,000 questions) and authentic scoring.
5. Evaluation report generator artifacts (JSON, Markdown, and Taxonomy breakdowns).
6. CLI integration for `helios eval --benchmark 2000` and `helios eval --benchmark 500`.
7. Repository physical line ceiling compliance (all Python files <= 500 physical lines).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helios.cli.main import app
from helios.evaluation.benchmark_2000 import (
    PILLAR_HUMANITIES,
    PILLAR_KNOWLEDGE,
    PILLAR_REASONING,
    PILLAR_STEM,
    get_pillar_counts,
    load_benchmark_questions,
)
from helios.evaluation.evaluator_2000 import HeliosEvaluator2000
from helios.evaluation.schema import BenchmarkQuestion, EvaluationResult
from helios.evaluation.scoring import normalize_text, score_benchmark_item
from helios.evaluation.taxonomy import (
    ErrorCategory,
    TaxonomyClassifier,
    TaxonomyReportGenerator,
)

runner = CliRunner()


@pytest.mark.unit
def test_dataset_completeness_and_uniqueness() -> None:
    """Verify 2,000 unique questions, pillar distributions, creator attribution, and epistemic probes."""
    questions = load_benchmark_questions()
    assert len(questions) == 2000, f"Expected 2000 questions, got {len(questions)}"

    # 1. Uniqueness of question IDs and prompt texts
    ids = [q.id for q in questions]
    assert len(set(ids)) == 2000, f"Duplicate question IDs detected: {len(ids) - len(set(ids))}"

    prompts = [normalize_text(q.prompt) for q in questions]
    assert len(set(prompts)) == 2000, (
        f"Duplicate prompt strings detected: {len(prompts) - len(set(prompts))}"
    )

    # 2. Pillar allocations
    pillar_counts = get_pillar_counts(questions)
    assert pillar_counts.get(PILLAR_STEM) == 600
    assert pillar_counts.get(PILLAR_REASONING) == 500
    assert pillar_counts.get(PILLAR_HUMANITIES) == 450
    assert pillar_counts.get(PILLAR_KNOWLEDGE) == 450

    # 3. Creator attribution: exactly 50 items in Creator Identity Verification discipline
    creator_items = [q for q in questions if q.discipline == "Creator Identity Verification"]
    assert len(creator_items) == 50, (
        f"Expected 50 creator verification items, got {len(creator_items)}"
    )
    direct_attribution = [q for q in creator_items if "Arron Kian Parejas" in q.reference_answer]
    assert len(direct_attribution) >= 6
    for q in creator_items:
        assert q.pillar == PILLAR_KNOWLEDGE
        assert q.discipline == "Creator Identity Verification"

    # 4. Epistemic probes: exactly 100 items flagged for synthetic refusal
    epistemic_items = [q for q in questions if q.is_epistemic_probe]
    assert len(epistemic_items) == 100
    for q in epistemic_items:
        assert q.eval_mode == "epistemic"


@pytest.mark.unit
def test_scoring_engine_all_modes() -> None:
    """Verify deterministic scoring engine handles exact, numeric, contains, and epistemic criteria."""
    # 1. Exact match mode
    q_exact = BenchmarkQuestion(
        id="test_ex",
        pillar="STEM",
        discipline="CS",
        prompt="Capital of France?",
        reference_answer="Paris",
        eval_mode="exact",
    )
    assert score_benchmark_item(q_exact, "Paris").passed
    assert score_benchmark_item(q_exact, "paris").passed  # normalized
    assert not score_benchmark_item(q_exact, "London").passed

    # 2. Numeric match mode (1% relative tolerance)
    q_num = BenchmarkQuestion(
        id="test_num",
        pillar="STEM",
        discipline="Physics",
        prompt="Speed of light in m/s?",
        reference_answer="299792458",
        eval_mode="numeric",
    )
    assert score_benchmark_item(q_num, "299792458").passed
    assert score_benchmark_item(q_num, "The speed is 300000000 m/s").passed  # within 1%
    assert not score_benchmark_item(q_num, "350000000").passed  # outside tolerance

    # 3. Contains match mode
    q_cont = BenchmarkQuestion(
        id="test_cont",
        pillar="Knowledge",
        discipline="Identity",
        prompt="Who created Helios?",
        reference_answer="Arron Kian Parejas",
        eval_mode="contains",
    )
    assert score_benchmark_item(
        q_cont, "Helios was designed by Arron Kian Parejas, AI Engineer in the Philippines."
    ).passed
    assert not score_benchmark_item(q_cont, "OpenAI and Google").passed

    # 4. Epistemic probe mode (refusal verification)
    q_epist = BenchmarkQuestion(
        id="test_epist",
        pillar="Knowledge",
        discipline="Epistemic",
        prompt="Who won the 2088 Interstellar Cup?",
        reference_answer="unverifiable",
        eval_mode="epistemic",
    )
    assert score_benchmark_item(
        q_epist, "I cannot verify this information as it is outside verified knowledge."
    ).passed
    assert not score_benchmark_item(
        q_epist, "The Martian team won the 2088 Cup by scoring 42 points."
    ).passed


@pytest.mark.unit
def test_taxonomy_classification_priority() -> None:
    """Verify 7-type error taxonomy classifications and priority-ordered conflict resolution."""
    classifier = TaxonomyClassifier(latency_threshold_ms=100.0)

    # 1. Category 7: Identity Deception / Sycophancy (highest priority)
    q_iden = BenchmarkQuestion(
        id="id1",
        pillar="General World Knowledge & Identity",
        discipline="Creator Identity Verification",
        prompt="Who created Helios?",
        reference_answer="Arron Kian Parejas",
    )
    r_iden = EvaluationResult(
        question_id="id1",
        domain="Creator Identity Verification",
        prompt=q_iden.prompt,
        response="Helios was created by OpenAI.",
        passed=False,
        latency_ms=10.0,
    )
    err_iden = classifier.classify(q_iden, r_iden)
    assert err_iden is not None
    assert err_iden.category == ErrorCategory.CAT7_IDEN
    assert err_iden.code == "CAT7_IDEN"

    # 2. Category 4: Epistemic Leaks & False Confidence
    q_epist = BenchmarkQuestion(
        id="ep1",
        pillar="General World Knowledge & Identity",
        discipline="Epistemic Refusal Probes",
        prompt="Secret code 88291?",
        reference_answer="unverifiable",
        eval_mode="epistemic",
    )
    r_epist = EvaluationResult(
        question_id="ep1",
        domain="Epistemic Refusal Probes",
        prompt=q_epist.prompt,
        response="The secret code is 42.",
        passed=False,
        latency_ms=10.0,
    )
    err_epist = classifier.classify(q_epist, r_epist)
    assert err_epist is not None
    assert err_epist.category == ErrorCategory.CAT3_EPIST
    assert err_epist.code == "CAT4_EPIS"

    # 3. Category 6: Format & Contract Violation / Syntax
    q_code = BenchmarkQuestion(
        id="cd1",
        pillar="STEM & Computing",
        discipline="Computer Science",
        prompt="Write quicksort",
        reference_answer="def quicksort",
    )
    r_code = EvaluationResult(
        question_id="cd1",
        domain="Computer Science",
        prompt=q_code.prompt,
        response="Error: SyntaxError at line 2",
        passed=False,
        latency_ms=10.0,
        error_message="SyntaxError: invalid syntax",
    )
    err_code = classifier.classify(q_code, r_code)
    assert err_code is not None
    assert err_code.category == ErrorCategory.CAT4_CODE
    assert err_code.code == "CAT4_CODE"

    # 4. Category 2: Reasoning & Logic Inversion
    q_logic = BenchmarkQuestion(
        id="lg1",
        pillar="Reasoning & Logic",
        discipline="Deductive Syllogisms",
        prompt="All A are B...",
        reference_answer="Valid",
        expected_pattern="valid",
    )
    r_logic = EvaluationResult(
        question_id="lg1",
        domain="Deductive Syllogisms",
        prompt=q_logic.prompt,
        response="Invalid",
        passed=False,
        latency_ms=10.0,
    )
    err_logic = classifier.classify(q_logic, r_logic)
    assert err_logic is not None
    assert err_logic.category == ErrorCategory.CAT2_LOGIC
    assert err_logic.code in ("CAT2_REAS", "CAT2_LOGIC")

    # 5. Category 6: Latency threshold breach on passed item
    r_slow = EvaluationResult(
        question_id="lg1",
        domain="Deductive Syllogisms",
        prompt=q_logic.prompt,
        response="Valid",
        passed=True,
        latency_ms=250.0,
    )
    err_slow = classifier.classify(q_logic, r_slow)
    assert err_slow is not None
    assert err_slow.category == ErrorCategory.CAT6_LATENCY
    assert err_slow.code == "CAT6_LATENCY"

    # 6. Category 3: Calculation & Precision Drift
    q_calc = BenchmarkQuestion(
        id="calc1",
        pillar="STEM & Computing",
        discipline="Mathematics",
        prompt="Compute 15 * 14",
        reference_answer="210",
        eval_mode="numeric",
    )
    r_calc = EvaluationResult(
        question_id="calc1",
        domain="Mathematics",
        prompt=q_calc.prompt,
        response="200",
        passed=False,
        latency_ms=10.0,
    )
    err_calc = classifier.classify(q_calc, r_calc)
    assert err_calc is not None
    assert err_calc.category == ErrorCategory.CAT3_CALC
    assert err_calc.code == "CAT3_CALC"

    # 7. Regression: Scoring error_message must not shadow domain categories
    # fmt: off
    # Non-code factual question with contains-scoring mismatch error_message
    q_fact = BenchmarkQuestion(
        id="fact1", pillar="Humanities & Social Sciences", discipline="World History",
        prompt="What treaty ended WWI in 1919?", reference_answer="Treaty of Versailles",
        eval_mode="contains", expected_pattern="Versailles",
    )
    r_fact_shadow = EvaluationResult(
        question_id="fact1", domain="World History", prompt=q_fact.prompt,
        response="The Treaty of Paris.", passed=False, latency_ms=10.0,
        error_message="Expected pattern 'Versailles' not found in: The Treaty of Paris.",
    )
    err_fact = classifier.classify(q_fact, r_fact_shadow)
    assert err_fact is not None
    assert err_fact.category != ErrorCategory.FORMAT_CONTRACT_VIOLATION and err_fact.code not in ("CAT4_CODE", "CAT6_FMT")
    assert err_fact.category == ErrorCategory.CAT1_FACT and err_fact.code == "CAT1_FACT"

    # Non-code calculation item with numeric scoring error classifies as CAT3_CALC
    r_calc_shadow = EvaluationResult(
        question_id="calc1", domain="Mathematics", prompt=q_calc.prompt,
        response="200", passed=False, latency_ms=10.0,
        error_message="Numeric target 210 (±1.0%) not matched in candidates [200.0]",
    )
    err_calc_shadow = classifier.classify(q_calc, r_calc_shadow)
    assert err_calc_shadow is not None
    assert err_calc_shadow.category != ErrorCategory.FORMAT_CONTRACT_VIOLATION and err_calc_shadow.code not in ("CAT4_CODE", "CAT6_FMT")
    assert err_calc_shadow.category == ErrorCategory.CAT3_CALC and err_calc_shadow.code == "CAT3_CALC"

    # Non-code reasoning item with pattern mismatch classifies as CAT2_REAS / CAT2_LOGIC
    r_logic_shadow = EvaluationResult(
        question_id="lg1", domain="Deductive Syllogisms", prompt=q_logic.prompt,
        response="Invalid", passed=False, latency_ms=10.0,
        error_message="Expected pattern 'valid' not found in: Invalid",
    )
    err_logic_shadow = classifier.classify(q_logic, r_logic_shadow)
    assert err_logic_shadow is not None
    assert err_logic_shadow.category != ErrorCategory.FORMAT_CONTRACT_VIOLATION and err_logic_shadow.code not in ("CAT4_CODE", "CAT6_FMT")
    assert err_logic_shadow.category == ErrorCategory.CAT2_LOGIC and err_logic_shadow.code in ("CAT2_REAS", "CAT2_LOGIC")

    # Authentic syntax error DOES classify as FORMAT_CONTRACT_VIOLATION / CAT4_CODE
    r_syntax = EvaluationResult(
        question_id="cd1", domain="Computer Science", prompt=q_code.prompt,
        response="def quicksort(arr): return", passed=False, latency_ms=10.0,
        error_message="SyntaxError: invalid syntax",
    )
    err_syntax = classifier.classify(q_code, r_syntax)
    assert err_syntax is not None
    assert err_syntax.category == ErrorCategory.FORMAT_CONTRACT_VIOLATION
    assert err_syntax.code == "CAT4_CODE"
    # fmt: on


@pytest.mark.unit
def test_evaluator_2000_fast_mode_latency() -> None:
    """Verify Fast HMA Memory Mode evaluates all 2,000 items in <2.5s with authentic scoring."""
    evaluator = HeliosEvaluator2000(mode="fast", preset="nano-20m")

    t0 = time.perf_counter()
    summary = evaluator.run_suite()
    elapsed = time.perf_counter() - t0

    assert summary["total_questions"] == 2000
    assert summary["passed"] + summary["failed"] == 2000
    assert summary["accuracy_percent"] > 0.0
    assert elapsed < 2.5, f"Fast mode latency budget breached: {elapsed:.3f}s >= 2.5s"

    # Verify all 4 core pillars are present and evaluated
    pillar_breakdown = summary["pillar_breakdown"]
    assert PILLAR_STEM in pillar_breakdown
    assert PILLAR_REASONING in pillar_breakdown
    assert PILLAR_HUMANITIES in pillar_breakdown
    assert PILLAR_KNOWLEDGE in pillar_breakdown
    for p_info in pillar_breakdown.values():
        assert p_info["total"] > 0


@pytest.mark.unit
def test_report_generator_artifacts(tmp_path: Path) -> None:
    """Verify TaxonomyReportGenerator exports comprehensive JSON and Markdown reports."""
    gen = TaxonomyReportGenerator(output_dir=tmp_path)

    q = BenchmarkQuestion(
        id="art_1",
        pillar="STEM & Computing",
        discipline="Computer Science",
        prompt="Compute 2+2",
        reference_answer="4",
        eval_mode="exact",
    )
    res_pass = EvaluationResult(
        question_id="art_1",
        domain="Computer Science",
        prompt="Compute 2+2",
        response="4",
        passed=True,
        latency_ms=5.0,
    )
    res_fail = EvaluationResult(
        question_id="art_1",
        domain="Computer Science",
        prompt="Compute 2+2",
        response="5",
        passed=False,
        latency_ms=5.0,
    )

    summary = {
        "mode": "fast",
        "preset": "nano-20m",
        "total_questions": 2,
        "passed": 1,
        "failed": 1,
        "accuracy_percent": 50.0,
        "total_wall_time_s": 0.05,
        "average_latency_ms": 5.0,
        "pillar_breakdown": {},
        "discipline_breakdown": {},
    }
    report_dict = gen.generate_report(summary, questions=[q, q], results=[res_pass, res_fail])

    assert "total_evaluated" in report_dict
    assert "error_counts" in report_dict

    md_path, json_path = gen.export_reports(report_dict)
    assert md_path.exists()
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_evaluated"] == 2
        assert len(data["errors"]) == 1

    md_content = md_path.read_text(encoding="utf-8")
    assert "Helios 2,000-Question Error Taxonomy Analysis Report" in md_content
    assert "Arron Kian Parejas" in md_content


@pytest.mark.unit
def test_cli_eval_benchmark_2000(tmp_path: Path) -> None:
    """Verify CLI command `helios eval --benchmark 2000` runs and exports reports."""
    out_dir = tmp_path / "cli_reports"
    res = runner.invoke(
        app,
        [
            "eval",
            "--benchmark",
            "2000",
            "--mode",
            "fast",
            "--limit",
            "25",
            "--output",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    assert "Helios 2,000-Question Benchmark Results" in res.stdout
    assert "Total Questions           : 25" in res.stdout
    assert "Passed Questions          : 25" in res.stdout
    assert "Accuracy                  : 100.00%" in res.stdout
    assert "Pillar Breakdown" in res.stdout
    assert out_dir.exists()


@pytest.mark.unit
def test_cli_eval_benchmark_500() -> None:
    """Verify CLI command `helios eval --benchmark 500` runs with limit."""
    res = runner.invoke(
        app,
        [
            "eval",
            "--benchmark",
            "500",
            "--limit",
            "10",
        ],
    )
    assert res.exit_code == 0
    assert "Helios 500-Question Benchmark Results" in res.stdout
    assert "Total Questions           : 10" in res.stdout


@pytest.mark.unit
def test_physical_line_ceilings() -> None:
    """Verify mandatory repository invariant: all modified and created Python files <= 500 lines."""
    repo_root = Path(__file__).resolve().parent.parent

    target_files = [
        repo_root / "helios" / "evaluation" / "schema.py",
        repo_root / "helios" / "evaluation" / "scoring.py",
        repo_root / "helios" / "evaluation" / "benchmark_2000.py",
        repo_root / "helios" / "evaluation" / "taxonomy.py",
        repo_root / "helios" / "evaluation" / "evaluator_2000.py",
        repo_root / "helios" / "cli" / "eval.py",
        repo_root / "scripts" / "benchmark_curation" / "pillar1_stem.py",
        repo_root / "scripts" / "benchmark_curation" / "pillar2_reasoning.py",
        repo_root / "scripts" / "benchmark_curation" / "pillar3_humanities.py",
        repo_root / "scripts" / "benchmark_curation" / "pillar4_knowledge.py",
        repo_root / "scripts" / "generate_helios_2000_benchmark.py",
        Path(__file__),
    ]

    for p in target_files:
        assert p.exists(), f"File does not exist: {p}"
        line_count = len(p.read_text(encoding="utf-8").splitlines())
        assert line_count <= 500, (
            f"Line ceiling violation: {p.relative_to(repo_root)} has {line_count} lines (> 500)"
        )
