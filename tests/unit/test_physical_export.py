"""Unit tests for physical model exporters (ONNX, TFLite, SafeTensors, TorchScript).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from helios.export.physical import (
    export_all_physical_formats,
    export_onnx,
    export_safetensors,
    export_tflite_binary,
    export_torchscript,
)
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM


@pytest.fixture
def mini_model_and_config():
    """Create lightweight Helios model for fast export tests."""
    cfg = HeliosConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        max_seq_len=32,
    )
    model = HeliosForCausalLM(cfg).cpu().eval()
    return model, cfg


def test_export_safetensors(mini_model_and_config):
    model, _ = mini_model_and_config
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "test.safetensors"
        bytes_written = export_safetensors(model, out_file)
        assert out_file.exists()
        assert bytes_written > 0
        assert out_file.stat().st_size == bytes_written


def test_export_torchscript(mini_model_and_config):
    model, _ = mini_model_and_config
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "test.pt"
        bytes_written = export_torchscript(model, out_file, seq_len=8)
        assert out_file.exists()
        assert bytes_written > 0
        loaded = torch.jit.load(str(out_file))
        dummy = torch.randint(0, 128, (1, 8), dtype=torch.long)
        out = loaded(dummy)
        assert out.shape == (1, 8, 128)


def test_export_tflite_binary(mini_model_and_config):
    model, cfg = mini_model_and_config
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "test.tflite"
        bytes_written = export_tflite_binary(model, cfg, out_file)
        assert out_file.exists()
        assert bytes_written > 0
        content = out_file.read_bytes()
        assert content[:4] == b"TFL3"


def test_export_onnx(mini_model_and_config):
    model, _ = mini_model_and_config
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "test.onnx"
        bytes_written = export_onnx(model, out_file, seq_len=8)
        assert out_file.exists()
        assert bytes_written > 0


def test_export_all_physical_formats(mini_model_and_config):
    model, cfg = mini_model_and_config
    with tempfile.TemporaryDirectory() as tmp_dir:
        res = export_all_physical_formats(model, cfg, output_dir=tmp_dir)
        assert "formats" in res
        for fmt in ("safetensors", "torchscript", "onnx", "tflite"):
            assert fmt in res["formats"]
            entry = res["formats"][fmt]
            assert Path(entry["file"]).exists()
            assert entry["bytes"] > 0
            assert len(entry["sha256"]) == 64
        assert Path(res["manifest_file"]).exists()
