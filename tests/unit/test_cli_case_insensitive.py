"""Unit Tests for Case-Insensitive CLI Subcommand Invocation (Feature 38).

Validates that all Helios CLI subcommands resolve and execute accurately
regardless of letter casing or hyphen/underscore formatting.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from helios.cli.main import app

runner = CliRunner()


@pytest.mark.unit
class TestCaseInsensitiveCLIInvocation:
    """Validate case-insensitive command resolution across all CLI subcommands."""

    @pytest.mark.parametrize(
        "subcommand",
        [
            "chat",
            "CHAT",
            "Chat",
            "eval",
            "EVAL",
            "Eval",
            "generate",
            "GENERATE",
            "Generate",
            "version",
            "VERSION",
            "Version",
            "inspect",
            "INSPECT",
            "Inspect",
            "roadmap",
            "ROADMAP",
            "Roadmap",
            "smoke-test",
            "SMOKE-TEST",
            "Smoke-Test",
            "resume-test",
            "RESUME-TEST",
            "Resume-Test",
            "auto-improve",
            "AUTO-IMPROVE",
            "Auto-Improve",
            "export",
            "EXPORT",
            "Export",
        ],
    )
    def test_all_subcommands_case_insensitive_help(self, subcommand: str) -> None:
        """Verify that every subcommand accepts uppercase and mixed-case --help."""
        result = runner.invoke(app, [subcommand, "--help"])
        assert result.exit_code == 0, (
            f"Failed for subcommand: {subcommand}\nOutput: {result.output}"
        )
        assert "Usage: " in result.stdout

    def test_version_command_casing_variations(self) -> None:
        """Verify execution of version command with diverse casing."""
        for cmd in ["version", "VERSION", "Version", "VeRsIoN"]:
            result = runner.invoke(app, [cmd])
            assert result.exit_code == 0
            assert "Helios version" in result.stdout
            assert "Arron Kian Parejas" in result.stdout

    def test_inspect_command_uppercase_execution(self) -> None:
        """Verify inspect command execution with uppercase subcommand."""
        result = runner.invoke(app, ["INSPECT", "-m", "nano-15m"])
        assert result.exit_code == 0
        assert "Helios Model Inspector" in result.stdout
        assert "Developer        : Arron Kian Parejas" in result.stdout

    def test_generate_command_uppercase_execution(self) -> None:
        """Verify generate command execution with uppercase subcommand."""
        result = runner.invoke(
            app,
            [
                "GENERATE",
                "--prompt",
                "def hello():",
                "--max-new-tokens",
                "4",
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
                "--no-stream",
            ],
        )
        assert result.exit_code == 0
        assert "Helios Generation Telemetry" in result.stdout

    def test_eval_command_uppercase_execution(self) -> None:
        """Verify eval command execution with uppercase subcommand."""
        result = runner.invoke(
            app,
            [
                "EVAL",
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
                "--prompt-len",
                "8",
                "--gen-len",
                "8",
            ],
        )
        assert result.exit_code == 0
        assert "Helios Performance & Perplexity Report" in result.stdout

    def test_chat_command_uppercase_execution(self) -> None:
        """Verify chat command accepts uppercase CHAT and exits cleanly on /exit."""
        result = runner.invoke(
            app,
            [
                "CHAT",
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
            ],
            input="/exit\n",
        )
        assert result.exit_code == 0
        assert "Helios Foundation Interactive Terminal Chat" in result.stdout
        assert "Exiting Helios Chat. Goodbye!" in result.stdout

    def test_nested_export_subcommands_casing(self) -> None:
        """Verify nested export commands accept uppercase group and subcommand names."""
        test_pairs = [
            ["EXPORT", "hf", "--help"],
            ["export", "HF", "--help"],
            ["EXPORT", "HF", "--help"],
            ["EXPORT", "OLLAMA", "--help"],
            ["EXPORT", "DISK", "--help"],
            ["Export", "Physical", "--help"],
        ]
        for pair in test_pairs:
            result = runner.invoke(app, pair)
            assert result.exit_code == 0, f"Failed for nested args: {pair}\nOutput: {result.output}"

    def test_hyphen_underscore_normalization(self) -> None:
        """Verify commands with hyphens also resolve when invoked with underscores."""
        for cmd in ["smoke_test", "SMOKE_TEST", "resume_test", "RESUME_TEST"]:
            result = runner.invoke(app, [cmd, "--help"])
            assert result.exit_code == 0, f"Failed for normalized cmd: {cmd}"

    def test_unknown_command_returns_code_2(self) -> None:
        """Verify that genuine unknown commands return exit code 2."""
        result = runner.invoke(app, ["UNKNOWN_CMD"])
        assert result.exit_code == 2
        assert "No such command" in result.output

    def test_pyproject_scripts_registration(self) -> None:
        """Verify pyproject.toml defines dual console scripts for helios and HELIOS."""
        pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        assert pyproject_path.exists()
        with pyproject_path.open("rb") as f:
            cfg = tomllib.load(f)
        scripts = cfg.get("project", {}).get("scripts", {})
        assert "helios" in scripts, "Missing lowercase 'helios' console script in pyproject.toml"
        assert "HELIOS" in scripts, "Missing uppercase 'HELIOS' console script in pyproject.toml"
        assert scripts["helios"] == scripts["HELIOS"]
