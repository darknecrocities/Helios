"""Tests for reasoning context optimizer, NPU export, duration parser, and chat commands."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from helios.cli.commands import (
    VALID_SLASH_COMMANDS,
    dispatch_chat_command,
    parse_slash_command,
)
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.reasoning.context_optimizer import (
    CONTEXT_PROFILES,
    ReasoningContextOptimizer,
    resolve_tier,
)
from helios_pipeline.npu.accelerator import (
    NPUAccelerator,
    detect_npu_backend,
    resolve_compute_device,
)
from helios_pipeline.workflow import parse_duration, run_doctor


class TestReasoningContextOptimizer:
    """Verifies dynamic NTK RoPE context scaling and attention sink compression."""

    def test_all_six_tiers_registered(self) -> None:
        expected = {"nano", "alpha", "beta", "core", "pro", "flagship"}
        assert set(CONTEXT_PROFILES.keys()) == expected
        assert CONTEXT_PROFILES["nano"].max_context == 16384
        assert CONTEXT_PROFILES["alpha"].max_context == 32768
        assert CONTEXT_PROFILES["beta"].max_context == 65536
        assert CONTEXT_PROFILES["core"].max_context == 131072
        assert CONTEXT_PROFILES["pro"].max_context == 131072
        assert CONTEXT_PROFILES["flagship"].max_context == 262144

    def test_ntk_scaling_factor(self) -> None:
        opt = ReasoningContextOptimizer("nano")
        # Within base training len (2048), scale factor is 1.0
        assert opt.ntk_scaling_factor(seq_len=2048) == 1.0
        # Beyond base context, scale factor dynamically expands > 1.0
        scale = opt.ntk_scaling_factor(seq_len=8192)
        assert scale > 1.0

    def test_kv_cache_sink_compression(self) -> None:
        opt = ReasoningContextOptimizer("nano")
        # B=1, H=2, S=32, D=16
        k = torch.randn(1, 2, 32, 16)
        v = torch.randn(1, 2, 32, 16)
        # Sinks: 4 tokens (default), window: 8 tokens -> total length 12
        k_comp, v_comp = opt.compress_kv_cache_with_sinks(k, v, window=8)
        assert k_comp.shape == (1, 2, 12, 16)
        assert v_comp.shape == (1, 2, 12, 16)
        # Exact identity of attention sinks preserved
        torch.testing.assert_close(k_comp[:, :, :4, :], k[:, :, :4, :])
        # Exact identity of recent rolling tokens preserved
        torch.testing.assert_close(k_comp[:, :, 4:, :], k[:, :, -8:, :])

    def test_pack_reasoning_context(self) -> None:
        opt = ReasoningContextOptimizer("alpha")
        packed = opt.pack_reasoning_context(
            system="System: You are Helios.",
            facts=["Fact 1: Helios is an open-weight model."],
            history=["User: Hi", "Assistant: Hello!"],
            query="User: Tell me about your architecture.",
        )
        assert "System: You are Helios." in packed
        assert "Fact 1: Helios is an open-weight model." in packed
        assert "User: Tell me about your architecture." in packed
        assert resolve_tier("alpha-100m") == "alpha"


class TestNPUAccelerator:
    """Verifies NPU hardware detection, device resolution, and ONNX export."""

    def test_detect_npu_backend(self) -> None:
        info = detect_npu_backend()
        assert info.backend in {"apple_ane", "ascend", "onnx_npu", "none"}
        assert info.train_device in {"mps", "npu", "cpu"}

    def test_resolve_compute_device(self) -> None:
        dev_auto = resolve_compute_device("auto")
        assert dev_auto in {"mps", "cuda", "cpu"}
        dev_cpu = resolve_compute_device("cpu")
        assert dev_cpu == "cpu"
        dev_npu = resolve_compute_device("npu")
        assert dev_npu in {"mps", "npu", "cpu"}

    def test_static_onnx_export(self) -> None:
        cfg = HeliosConfig.nano_15m()
        cfg.vocab_size = 256
        cfg.max_seq_len = 32
        model = HeliosForCausalLM(cfg)
        model.eval()

        accel = NPUAccelerator()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "npu_export"
            exported = accel.export_npu_package(
                model=model, vocab_size=cfg.vocab_size, out_dir=out_dir, seq_len=16
            )
            assert exported.exists()
            assert exported.stat().st_size > 0
            assert (out_dir / "npu_manifest.txt").exists()


class TestWorkflowAndStorageHardLock:
    """Verifies duration parsing, OneTouch hard-lock, and doctor diagnostics."""

    def test_parse_duration(self) -> None:
        assert parse_duration("5m") == 5.0
        assert parse_duration("90s") == 1.5
        assert parse_duration("2h") == 120.0
        assert parse_duration("3.5") == 3.5
        assert parse_duration(None, default_minutes=10.0) == 10.0

    def test_parse_duration_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration("invalid_duration")

    def test_run_doctor_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if not Path("/Volumes/One Touch").exists():
            mock_sm = MagicMock()
            mock_sm.return_value.check_health.return_value = {
                "mount_path": "/Volumes/One Touch",
                "free_gb": 850.0,
                "total_gb": 1000.0,
                "helios_usage_gb": 12.5,
                "status": "HEALTHY",
            }
            monkeypatch.setattr("helios_pipeline.storage.manager.StorageManager", mock_sm)
        exit_code = run_doctor()
        assert exit_code == 0


class TestChatSlashCommands:
    """Verifies slash command routing, tokenization, and help text."""

    def test_slash_commands_registered(self) -> None:
        required = {"/train", "/bench", "/sync", "/doctor"}
        assert required.issubset(VALID_SLASH_COMMANDS)

    def test_parse_slash_commands(self) -> None:
        assert parse_slash_command("/train nano -m 5") == ("/train", "nano -m 5")
        assert parse_slash_command("/bench all") == ("/bench", "all")
        assert parse_slash_command("/sync git") == ("/sync", "git")
        assert parse_slash_command("/doctor") == ("/doctor", "")
        assert parse_slash_command("plain text prompt") is None

    def test_dispatch_doctor_command(self) -> None:
        mock_session = MagicMock()
        mock_session.preset = "helios-nano-20m"
        res = dispatch_chat_command("/doctor", mock_session)
        assert res.handled is True
        assert res.should_exit is False
