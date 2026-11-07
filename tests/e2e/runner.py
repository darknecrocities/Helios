"""Helios Standalone E2E Test Suite Runner.

Executes requirement-driven opaque-box test suites across Tiers 1-4.
Provides formatted CLI summary tables, per-tier duration metrics,
exit code semantics (0=pass, 1=fail, 2=config error), and JSON/Markdown report exports.

Usage:
    python -m tests.e2e.runner [--tier {1,2,3,4,all}] [-k KEYWORD] [-x]
    python tests/e2e/runner.py --json-report results.json --markdown-report report.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

import helios


@dataclass
class TestCaseResult:
    """Individual test outcome record."""

    node_id: str
    name: str
    tier: str
    outcome: str  # "passed" | "failed" | "skipped" | "error"
    duration_seconds: float
    error_message: str | None = None


@dataclass
class TierSummary:
    """Summary metrics for a specific test tier."""

    tier_name: str
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    duration_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.passed / self.total) * 100.0


@dataclass
class TestRunReport:
    """Overall test run execution report."""

    timestamp: str
    helios_version: str
    author: str
    device: str
    tier_filter: str
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    total_duration_seconds: float = 0.0
    tiers: dict[str, TierSummary] = field(default_factory=dict)
    cases: list[TestCaseResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0 and self.error_tests == 0 and self.total_tests > 0


class E2EReporterPlugin:
    """Pytest plugin collecting structured execution metrics per tier."""

    def __init__(self, target_tier: str) -> None:
        self.target_tier = target_tier
        self.results: list[TestCaseResult] = []
        self.start_time = 0.0
        self.end_time = 0.0

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        self.start_time = time.time()

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        self.end_time = time.time()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # Only capture the call phase or failed setup/teardown phases
        if report.when == "call" or (report.when in ("setup", "teardown") and report.failed):
            node_id = report.nodeid
            name = report.location[2] if report.location else node_id

            # Determine tier from markers or node_id path
            tier = "unknown"
            if "tier1" in node_id or "test_tier1" in node_id:
                tier = "tier1"
            elif "tier2" in node_id or "test_tier2" in node_id:
                tier = "tier2"
            elif "tier3" in node_id or "test_tier3" in node_id:
                tier = "tier3"
            elif "tier4" in node_id or "test_tier4" in node_id:
                tier = "tier4"

            outcome = report.outcome
            if report.failed and report.when != "call":
                outcome = "error"

            err_msg = None
            if report.failed:
                err_msg = str(report.longrepr)

            self.results.append(
                TestCaseResult(
                    node_id=node_id,
                    name=name,
                    tier=tier,
                    outcome=outcome,
                    duration_seconds=report.duration,
                    error_message=err_msg,
                )
            )


def build_run_report(plugin: E2EReporterPlugin, target_tier: str, device: str) -> TestRunReport:
    """Synthesize collected plugin outcomes into a structured TestRunReport."""
    duration = plugin.end_time - plugin.start_time
    report = TestRunReport(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        helios_version=getattr(helios, "__version__", "0.1.0"),
        author="Arron Kian Parejas <parejasarronkian@gmail.com>",
        device=device,
        tier_filter=target_tier,
        total_duration_seconds=duration,
    )

    tier_map: dict[str, TierSummary] = {
        "tier1": TierSummary("Tier 1: Feature Equivalence"),
        "tier2": TierSummary("Tier 2: Boundary & Limits"),
        "tier3": TierSummary("Tier 3: Pairwise Combinations"),
        "tier4": TierSummary("Tier 4: Real-World Scenarios"),
    }

    for case in plugin.results:
        report.total_tests += 1
        t_summary = tier_map.setdefault(case.tier, TierSummary(case.tier))
        t_summary.total += 1
        t_summary.duration_seconds += case.duration_seconds

        if case.outcome == "passed":
            report.passed_tests += 1
            t_summary.passed += 1
        elif case.outcome == "failed":
            report.failed_tests += 1
            t_summary.failed += 1
        elif case.outcome == "skipped":
            report.skipped_tests += 1
            t_summary.skipped += 1
        elif case.outcome == "error":
            report.error_tests += 1
            t_summary.errors += 1

        report.cases.append(case)

    # Filter out empty tiers
    report.tiers = {k: v for k, v in tier_map.items() if v.total > 0}
    return report


def render_cli_summary(report: TestRunReport) -> None:
    """Print high-contrast, informative ANSI terminal summary table."""
    cyan = "\033[96m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"
    bold = "\033[1m"
    reset = "\033[0m"

    print("\n" + "=" * 78)
    print(f"{bold}{cyan}HELIOS FOUNDATION MODEL — OPAQUE-BOX E2E TEST RUNNER{reset}")
    print(f"Developer    : {bold}{report.author}{reset}")
    v_info = f"{bold}{report.helios_version}{reset} | Device: {bold}{report.device}{reset}"
    print(f"Version      : {v_info}")
    print(f"Target Filter: {bold}{report.tier_filter.upper()}{reset}")
    print("=" * 78)

    hdr = f"{'Tier Name':<32} {'Total':>7} {'Pass':>7} {'Fail':>7} {'Skip':>7} {'Time (s)':>10}"
    print(f"{bold}{hdr}{reset}")
    print("-" * 78)

    for _, s in report.tiers.items():
        status_color = green if s.failed == 0 and s.errors == 0 else red
        print(
            f"{s.tier_name:<32} {s.total:>7} "
            f"{green}{s.passed:>7}{reset} "
            f"{status_color}{s.failed + s.errors:>7}{reset} "
            f"{yellow}{s.skipped:>7}{reset} "
            f"{s.duration_seconds:>10.2f}"
        )

    print("-" * 78)
    status_label = f"{green}PASSED (100%){reset}" if report.all_passed else f"{red}FAILED{reset}"
    fail_color = red if report.failed_tests + report.error_tests > 0 else green
    total_fails = report.failed_tests + report.error_tests
    print(
        f"{bold}{'TOTAL / VERDICT':<32} {report.total_tests:>7} "
        f"{green}{report.passed_tests:>7}{reset} "
        f"{fail_color}{total_fails:>7}{reset} "
        f"{yellow}{report.skipped_tests:>7}{reset} "
        f"{report.total_duration_seconds:>10.2f}s  [{status_label}]"
    )
    print("=" * 78 + "\n")


def export_reports(
    report: TestRunReport,
    json_path: Path | None = None,
    markdown_path: Path | None = None,
) -> None:
    """Export test results to machine-parseable JSON and/or formatted Markdown."""
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(report)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Exported JSON test report to: {json_path}")

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        md_lines = [
            f"# Helios E2E Test Execution Report — {report.timestamp}",
            f"- **Author & Lead**: {report.author}",
            f"- **Helios Version**: {report.helios_version}",
            f"- **Execution Device**: `{report.device}`",
            f"- **Total Duration**: {report.total_duration_seconds:.2f} seconds",
            f"- **Overall Status**: **{'PASSED (100%)' if report.all_passed else 'FAILED'}**",
            "",
            "## Summary Table",
            "| Tier | Total | Passed | Failed | Skipped | Pass Rate | Time |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _, s in report.tiers.items():
            md_lines.append(
                f"| **{s.tier_name}** | {s.total} | {s.passed} | {s.failed + s.errors} | "
                f"{s.skipped} | {s.pass_rate:.1f}% | {s.duration_seconds:.2f}s |"
            )
        md_lines.append(
            f"| **TOTAL** | **{report.total_tests}** | **{report.passed_tests}** | "
            f"**{report.failed_tests + report.error_tests}** | **{report.skipped_tests}** | "
            f"**{(report.passed_tests / max(1, report.total_tests)) * 100:.1f}%** | "
            f"**{report.total_duration_seconds:.2f}s** |"
        )
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"✓ Exported Markdown test report to: {markdown_path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI runner invocation options."""
    parser = argparse.ArgumentParser(
        description="Helios Open-Weight LLM: Standalone Opaque-Box E2E Test Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "all"],
        help="Select specific test tier or 'all' for complete Tiers 1-4 suite.",
    )
    parser.add_argument(
        "-k",
        "--keyword",
        type=str,
        default=None,
        help="Pytest keyword expression to filter test execution.",
    )
    parser.add_argument(
        "-x",
        "--fail-fast",
        action="store_true",
        help="Stop test execution immediately upon first failure.",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=None,
        help="Path to export structured JSON results.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help="Path to export human-readable Markdown summary report.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose pytest output.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to assign for model test runs ('cpu', 'mps', 'auto').",
    )
    return parser.parse_args()


def main() -> int:
    """Runner entry point with exit code semantics: 0=pass, 1=fail, 2=arg error."""
    try:
        args = parse_args()
    except SystemExit as exc:
        return 2 if exc.code != 0 else 0

    e2e_dir = Path(__file__).resolve().parent

    pytest_args = ["-q", "--no-header"]
    if args.fail_fast:
        pytest_args.append("-x")
    if args.verbose:
        pytest_args.append("-v")
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])

    # Map tier argument to test files and markers
    if args.tier == "1":
        pytest_args.append(str(e2e_dir / "test_tier1_features.py"))
    elif args.tier == "2":
        pytest_args.append(str(e2e_dir / "test_tier2_boundaries.py"))
    elif args.tier == "3":
        pytest_args.append(str(e2e_dir / "test_tier3_combinations.py"))
    elif args.tier == "4":
        pytest_args.append(str(e2e_dir / "test_tier4_scenarios.py"))
    else:
        pytest_args.append(str(e2e_dir))

    plugin = E2EReporterPlugin(target_tier=args.tier)
    pytest.main(pytest_args, plugins=[plugin])

    report = build_run_report(plugin, target_tier=args.tier, device=args.device)
    render_cli_summary(report)
    export_reports(report, json_path=args.json_report, markdown_path=args.markdown_report)

    # Return 0 if all tests passed, 1 if any failure
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
