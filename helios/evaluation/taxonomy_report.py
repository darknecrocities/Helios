"""Taxonomy report generator and export engine for Helios.

Aggregates benchmark evaluation failures, computes failure distributions,
and generates structured Markdown and JSON remediation reports.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helios.evaluation.schema import BenchmarkQuestion, EvaluationResult

if TYPE_CHECKING:
    from helios.evaluation.taxonomy import ClassifiedError, TaxonomyClassifier


class TaxonomyReportGenerator:
    """Aggregates benchmark evaluation failures and generates comprehensive remediation reports."""

    def __init__(
        self,
        classifier: TaxonomyClassifier | None = None,
        output_dir: Path | str | None = None,
    ) -> None:
        if classifier is None:
            from helios.evaluation.taxonomy import TaxonomyClassifier

            self.classifier = TaxonomyClassifier()
        else:
            self.classifier = classifier
        self.output_dir = Path(output_dir).resolve() if output_dir else Path("reports").resolve()

    def generate_report(
        self,
        results_or_summary: list[EvaluationResult] | dict[str, Any],
        questions: list[BenchmarkQuestion] | None = None,
        model_name: str = "Helios Foundation (nano-20m)",
        results: list[EvaluationResult] | None = None,
    ) -> dict[str, Any]:
        """Aggregate results and classify failures into the 7-Type Taxonomy."""
        from helios.evaluation.taxonomy import (
            CATEGORY_CODES,
            ErrorCategory,
        )

        if isinstance(results_or_summary, dict):
            summary = results_or_summary
            eval_results = results or []
            model_name = summary.get("preset", summary.get("model_name", model_name))
        else:
            summary = {}
            eval_results = results_or_summary or []

        q_map = {q.id: q for q in questions} if questions else {}
        total = summary.get("total_questions", len(eval_results))
        passed = summary.get("passed", sum(1 for r in eval_results if r.passed))
        failed = summary.get("failed", total - passed)
        acc = summary.get("accuracy_percent", (passed / total * 100.0) if total > 0 else 0.0)

        classified_errors: list[ClassifiedError] = []
        for r in eval_results:
            if not r.passed or (r.latency_ms > self.classifier.latency_threshold_ms):
                q = q_map.get(r.question_id)
                if q is None:
                    q = BenchmarkQuestion(
                        id=r.question_id,
                        discipline=r.domain,
                        question=r.prompt,
                        expected_pattern="",
                        reference_answer="",
                        explanation="",
                    )
                err = self.classifier.classify(q, r)
                classified_errors.append(err)

        category_counts = {cat: 0 for cat in ErrorCategory}
        for err in classified_errors:
            category_counts[err.category] += 1

        category_breakdown = {
            cat.value: {
                "code": CATEGORY_CODES.get(cat, cat.name),
                "count": cnt,
                "percent_of_failures": (cnt / len(classified_errors) * 100.0)
                if classified_errors
                else 0.0,
            }
            for cat, cnt in category_counts.items()
        }

        return {
            "model_name": model_name,
            "total_questions": total,
            "total_evaluated": total,
            "passed": passed,
            "failed": failed,
            "accuracy_percent": acc,
            "category_breakdown": category_breakdown,
            "error_counts": {info["code"]: info["count"] for info in category_breakdown.values()},
            "errors": [err.to_dict() for err in classified_errors],
        }

    def export_reports(
        self, report_data: dict[str, Any], output_dir: Path | str | None = None
    ) -> tuple[Path, Path]:
        """Export both Markdown and JSON reports to output_dir, returning (md_path, json_path)."""
        target_dir = Path(output_dir).resolve() if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        md_path = self.export_markdown(
            report_data, output_path=target_dir / "benchmark_2000_report.md"
        )
        json_path = self.export_json(
            report_data, output_path=target_dir / "benchmark_2000_errors.json"
        )
        return md_path, json_path

    def export_markdown(
        self,
        report_data: dict[str, Any],
        output_path: Path | str = "reports/benchmark_2000_report.md",
    ) -> Path:
        """Export human-readable Markdown report with executive summary and error tables."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Helios 2,000-Question Error Taxonomy Analysis Report",
            "## Automated Error Taxonomy, Failure Remediation & Closed-Loop Regression Analysis",
            "",
            "**Architect & Lead Developer**: Arron Kian Parejas (`parejasarronkian@gmail.com`)  ",
            f"**Model Evaluated**: {report_data.get('model_name', 'Helios Foundation')}  ",
            f"**Total Questions**: {report_data['total_questions']}  ",
            f"**Passed**: {report_data['passed']}  ",
            f"**Failed**: {report_data['failed']}  ",
            f"**Overall Benchmark Accuracy**: {report_data['accuracy_percent']:.2f}%  ",
            "",
            "## 1. 7-Type Error Taxonomy Breakdown",
            "",
            "| Code | Category | Failures | % of Total Failures |",
            "| :---: | :--- | :---: | :---: |",
        ]
        for cat_name, info in report_data["category_breakdown"].items():
            lines.append(
                f"| {info['code']} | {cat_name} | {info['count']} | {info['percent_of_failures']:.1f}% |"
            )

        lines.extend(
            [
                "",
                "## 2. Methodology & Verification Invariants",
                "- **7-Type Error Taxonomy**: Every failure is deterministically triaged across Factual Hallucination, Reasoning Inversion, Calculation Drift, Epistemic Leaks, Language Drift, Format Violations, and Identity Deception.",
                "- **Closed-Loop Regression**: Identified failures are permanently captured in targeted pytest suites (`tests/test_benchmark_2000.py`).",
                "- **Zero Mock Mandate**: All evaluation results reflect authentic neural generation and deterministic regex scoring.",
            ]
        )

        if report_data["errors"]:
            lines.extend(
                [
                    "",
                    "## 3. Failure Triage & Remediation Actions",
                    "",
                    "| ID | Code | Prompt | Root Cause | Remediation Action |",
                    "| :--- | :---: | :--- | :--- | :--- |",
                ]
            )
            for err in report_data["errors"][:50]:
                clean_p = err["prompt"].replace("\n", " ")[:40]
                lines.append(
                    f"| {err['question_id']} | {err['code']} | {clean_p}... | {err['root_cause']} | {err['remediation_action']} |"
                )
        else:
            lines.extend(
                [
                    "",
                    "## 3. Failure Triage & Remediation Actions",
                    "",
                    "**Zero failures observed.** All evaluated benchmark questions satisfied exact validation criteria.",
                ]
            )

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def export_json(
        self,
        report_data: dict[str, Any],
        output_path: Path | str = "reports/benchmark_2000_errors.json",
    ) -> Path:
        """Export machine-readable JSON evaluation and failure report."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
        return out
