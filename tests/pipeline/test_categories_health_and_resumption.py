"""Tests for category selection, checkpoint health/corruption audit, SOT dataset docs sync,
statistical sync summaries, and new `helios chat` slash commands.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from helios.cli.commands import VALID_SLASH_COMMANDS, dispatch_chat_command
from helios_pipeline.data.categories import (
    CATEGORY_SPECS,
    get_category_catalog,
    matches_category,
    normalize_category,
)
from helios_pipeline.data.docs_updater import DatasetDocumentationUpdater
from helios_pipeline.evaluation.health import (
    compute_file_sha256,
    inspect_tier_checkpoint,
)
from helios_pipeline.sync.orchestrator import build_sync_statistics_summary


class TestPretrainingCategories:
    """Verifies all 10 pretraining categories, aliases, and SOT dataset catalog mapping."""

    def test_normalize_categories_and_aliases(self) -> None:
        assert len(CATEGORY_SPECS) == 10
        assert normalize_category("math") == "math"
        assert normalize_category("fineweb") == "fineweb"
        assert normalize_category("web") == "fineweb"
        assert normalize_category("hist") == "history"
        assert normalize_category("code") == "coding"
        assert normalize_category("mcp") == "agents"
        assert normalize_category("filipino") == "tagalog"
        with pytest.raises(ValueError, match="Unknown pretraining category"):
            normalize_category("nonexistent_domain")

    def test_matches_category(self) -> None:
        math_rec = {"text": "Solve the quadratic equation x^2 - 5x + 6 = 0 using algebra."}
        hist_rec = {
            "text": "The Roman Empire expanded across the Mediterranean in classical antiquity."
        }
        assert matches_category(math_rec, "math") is True
        assert matches_category(hist_rec, "history") is True
        assert matches_category(math_rec, "all") is True

    def test_get_category_catalog_counts_177_datasets(self) -> None:
        catalog = get_category_catalog("sot/datasets_sot.json")
        assert len(catalog) == 10
        all_entry = next(c for c in catalog if c["key"] == "all")
        assert all_entry["dataset_count"] == 177
        assert all_entry["records_count"] == 120616


class TestCheckpointHealthAndCorruptionAudit:
    """Verifies HEALTHY, CORRUPTED (NaN/Inf & SHA-256 mismatch), and improvement deltas."""

    def test_uninitialized_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = inspect_tier_checkpoint("nano", base_dir=Path(tmp))
            assert rep.status == "UNINITIALIZED"

    def test_healthy_checkpoint_and_improvement_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            latest = base / "checkpoints" / "nano_continuous" / "latest"
            latest.mkdir(parents=True)
            weights = {"w1": torch.ones(4, 4), "w2": torch.full((4,), 0.5)}
            st_file = latest / "model.safetensors"
            save_file(weights, str(st_file))
            sha = compute_file_sha256(st_file)
            (latest / "SHA256SUMS").write_text(f"{sha}  model.safetensors\n", encoding="utf-8")
            manifest = {
                "tier": "nano",
                "mode": "full_parameter",
                "category": "math",
                "datasets_synced": 177,
                "resumed_from_step": 100,
                "total_steps": 250,
                "baseline_initial_loss": 10.0,
                "initial_loss": 5.0,
                "previous_loss": 5.0,
                "final_loss": 2.5,
                "previous_ppl": 20.0,
                "heldout_ppl": 5.0,
                "train_tok_per_s": 3100.0,
                "infer_tok_per_s": 8200.0,
                "sha256_digest": sha,
            }
            (latest / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            rep = inspect_tier_checkpoint("nano", base_dir=base)
            assert rep.status == "HEALTHY"
            assert rep.sha256_verified is True
            assert rep.nan_tensors == 0
            assert rep.inf_tensors == 0
            assert rep.category == "math"
            assert rep.resumed_from_step == 100
            assert rep.total_steps == 250
            assert rep.loss_improvement_pct == 75.0
            assert rep.ppl_reduction_pct == 75.0

    def test_corrupted_on_nan_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            latest = base / "checkpoints" / "nano_continuous" / "latest"
            latest.mkdir(parents=True)
            weights = {"w1": torch.tensor([1.0, float("nan"), 3.0])}
            st_file = latest / "model.safetensors"
            save_file(weights, str(st_file))
            sha = compute_file_sha256(st_file)
            (latest / "SHA256SUMS").write_text(f"{sha}  model.safetensors\n", encoding="utf-8")
            rep = inspect_tier_checkpoint("nano", base_dir=base)
            assert rep.status == "CORRUPTED"
            assert rep.nan_tensors == 1

    def test_corrupted_on_sha256_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            latest = base / "checkpoints" / "nano_continuous" / "latest"
            latest.mkdir(parents=True)
            st_file = latest / "model.safetensors"
            save_file({"w1": torch.ones(2, 2)}, str(st_file))
            (latest / "manifest.json").write_text(
                json.dumps({"sha256_digest": "0" * 64}), encoding="utf-8"
            )
            rep = inspect_tier_checkpoint("nano", base_dir=base)
            assert rep.status == "CORRUPTED"
            assert rep.sha256_verified is False


class TestDocsSyncAndChatCommands:
    """Verifies 177-dataset docs sync, statistical summary builder, and chat commands."""

    def test_sync_sot_datasets_to_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            updater = DatasetDocumentationUpdater(docs_dir=tmp)
            info = updater.sync_sot_datasets(
                sot_path="sot/datasets_sot.json", active_category="fineweb", tokens_trained=50000
            )
            assert info["total_datasets"] == 177
            assert info["total_records"] == 120616
            text = (Path(tmp) / "DATASET_CATALOG.md").read_text(encoding="utf-8")
            assert "177" in text
            assert "FineWeb-Edu" in text

    def test_build_sync_statistics_summary(self) -> None:
        headline, md = build_sync_statistics_summary()
        assert "sync(pretrained):" in headline
        assert "177 datasets" in headline
        assert "HELIOS Pretraining Telemetry & Health Sync" in md

    def test_new_chat_slash_commands_registered_and_routed(self) -> None:
        for cmd in ("/status", "/health", "/categories", "/publish", "/update"):
            assert cmd in VALID_SLASH_COMMANDS

        mock_session = MagicMock()
        mock_session.preset = "nano-20m"
        assert dispatch_chat_command("/categories", mock_session).handled is True
        assert dispatch_chat_command("/status", mock_session).handled is True
        assert dispatch_chat_command("/health", mock_session).handled is True
