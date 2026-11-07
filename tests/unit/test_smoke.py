"""Smoke and sanity tests for Helios package installation and CLI."""

import pytest
from typer.testing import CliRunner

import helios
from helios.cli.main import app

runner = CliRunner()


@pytest.mark.unit
def test_package_version() -> None:
    """Verify that helios exposes the expected version string."""
    assert helios.__version__ == "0.1.0"


@pytest.mark.unit
def test_cli_version_command() -> None:
    """Verify that the helios CLI version command executes successfully."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Helios version 0.1.0" in result.stdout


@pytest.mark.unit
def test_cli_inspect_command() -> None:
    """Verify that the helios CLI inspect command executes with default and custom options."""
    result_default = runner.invoke(app, ["inspect"])
    assert result_default.exit_code == 0
    assert "Helios Model Inspector - Target: nano-20m" in result_default.stdout

    result_custom = runner.invoke(app, ["inspect", "--model-size", "nano-35m"])
    assert result_custom.exit_code == 0
    assert "Helios Model Inspector - Target: nano-35m" in result_custom.stdout


@pytest.mark.unit
def test_cli_help() -> None:
    """Verify that the helios CLI help menu displays properly."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Helios Open-Weight LLM CLI" in result.stdout
