"""Unit tests for Git staging safety validator and secret blocking."""

from pathlib import Path
from tempfile import TemporaryDirectory

from helios_pipeline.git_sync.validator import (
    StagingValidator,
)


def test_validator_blocks_binary_weights() -> None:
    """Ensure weights (.pt, .safetensors, .bin, .onnx) are strictly rejected from git staging."""
    validator = StagingValidator()

    for ext in [".pt", ".pth", ".bin", ".safetensors", ".onnx", ".parquet"]:
        fake_file = Path(f"model_weights{ext}")
        allowed, reason = validator.is_file_allowed(fake_file)
        assert not allowed
        assert "Prohibited binary/weight extension" in reason


def test_validator_blocks_blocked_directories() -> None:
    """Ensure datasets and checkpoints folders cannot be staged to git."""
    validator = StagingValidator()

    bad_paths = [
        Path("datasets/raw/shard_001.json"),
        Path("checkpoints/step_00000100/state.pt"),
        Path(".tmp_stage/file.txt"),
    ]
    for p in bad_paths:
        allowed, reason = validator.is_file_allowed(p)
        assert not allowed
        assert "resides in blocked directory" in reason


def test_validator_blocks_credentials_and_secrets() -> None:
    """Ensure API keys, tokens, and private keys are flagged and blocked."""
    validator = StagingValidator()

    with TemporaryDirectory() as tmp_dir:
        secret_file = Path(tmp_dir) / "config.py"
        fake_token = "hf_" + "x" * 35
        secret_file.write_text(f'HF_TOKEN = "{fake_token}"\n', encoding="utf-8")
        allowed, reason = validator.is_file_allowed(secret_file)
        assert not allowed
        assert "Potential credential/secret detected" in reason


def test_validator_permits_safe_markdown_and_code() -> None:
    """Ensure regular documentation and clean code files are permitted."""
    validator = StagingValidator()

    with TemporaryDirectory() as tmp_dir:
        safe_file = Path(tmp_dir) / "DATASET_CATALOG.md"
        safe_file.write_text("# HELIOS Dataset Catalog\nClean documentation.\n", encoding="utf-8")
        allowed, reason = validator.is_file_allowed(safe_file)
        assert allowed
        assert reason == "Safe to commit"
