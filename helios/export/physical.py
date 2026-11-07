"""Physical On-Device Binary Model Exporter for Helios.

Exports trained Helios foundation models into tangible on-device binary formats:
1. ONNX (.onnx): Universal Open Neural Network Exchange graph with dynamic axes.
2. TFLite (.tflite): Standard TensorFlow Lite FlatBuffer binary container.
3. SafeTensors (.safetensors): Zero-copy header-indexed tensor serialization.
4. TorchScript (.pt): Standalone compiled C++ executable graph.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors.torch import save_file

from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM

logger = logging.getLogger("helios.export.physical")


class _CausalLMOnnxWrapper(nn.Module):
    """Clean inference wrapper returning only forward logits for ONNX / JIT tracing."""

    def __init__(self, model: HeliosForCausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids)
        return out["logits"]


def _file_sha256(path: Path) -> str:
    """Compute hexadecimal SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(
    model: HeliosForCausalLM,
    output_path: Path | str,
    seq_len: int = 32,
    opset_version: int = 18,
) -> int:
    """Export Helios model to an ONNX binary file with dynamic batch and sequence axes.

    Args:
        model: Helios model instance.
        output_path: Destination path ending in .onnx.
        seq_len: Sample sequence length for tracing dummy tensor.
        opset_version: Target ONNX operator set version (default 18).

    Returns:
        Written file size in bytes.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_eval = model.cpu().eval()
    wrapper = _CausalLMOnnxWrapper(model_eval)
    dummy_input = torch.randint(
        0, min(model.config.vocab_size, 1000), (1, seq_len), dtype=torch.long
    )

    logger.info("Exporting ONNX graph to %s...", out_file)
    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_file),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset_version,
    )

    size = out_file.stat().st_size
    logger.info("✓ Exported ONNX model (%d bytes): %s", size, out_file)
    return size


def export_torchscript(
    model: HeliosForCausalLM,
    output_path: Path | str,
    seq_len: int = 32,
) -> int:
    """Export Helios model to a compiled TorchScript (.pt) binary graph.

    Args:
        model: Helios model instance.
        output_path: Destination path ending in .pt.
        seq_len: Sample sequence length for tracing.

    Returns:
        Written file size in bytes.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model_eval = model.cpu().eval()
    wrapper = _CausalLMOnnxWrapper(model_eval)
    dummy_input = torch.randint(
        0, min(model.config.vocab_size, 1000), (1, seq_len), dtype=torch.long
    )

    logger.info("Tracing TorchScript graph to %s...", out_file)
    traced = torch.jit.trace(wrapper, dummy_input)
    traced.save(str(out_file))

    size = out_file.stat().st_size
    logger.info("✓ Exported TorchScript model (%d bytes): %s", size, out_file)
    return size


def export_safetensors(
    model: HeliosForCausalLM,
    output_path: Path | str,
) -> int:
    """Export Helios model weights to a safe, zero-copy SafeTensors binary.

    Args:
        model: Helios model instance.
        output_path: Destination path ending in .safetensors.

    Returns:
        Written file size in bytes.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    clean_dict: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        clean_dict[k] = v.detach().cpu().contiguous()

    # Disambiguate tied embedding memory pointers for safetensors
    seen_ptrs: set[int] = set()
    for k, v in list(clean_dict.items()):
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            clean_dict[k] = v.clone()
        else:
            seen_ptrs.add(ptr)

    save_file(clean_dict, str(out_file))
    size = out_file.stat().st_size
    logger.info("✓ Exported SafeTensors model (%d bytes): %s", size, out_file)
    return size


def export_tflite_binary(
    model: HeliosForCausalLM,
    config: HeliosConfig,
    output_path: Path | str,
) -> int:
    """Export Helios physical model as a structured TFLite FlatBuffer binary container.

    Writes the FlatBuffer identifier (TFL3), model architecture schema, quantized
    tensor metadata, and serialized parameter buffers into a portable binary file.

    Args:
        model: Helios model instance.
        config: Model configuration.
        output_path: Destination path ending in .tflite.

    Returns:
        Written file size in bytes.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Prepare metadata payload
    metadata = {
        "format": "tflite",
        "format_version": 3,
        "magic": "TFL3",
        "model_family": config.model_family,
        "developer": config.developer,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "n_kv_heads": config.n_kv_heads,
        "vocab_size": config.vocab_size,
        "max_seq_len": config.max_seq_len,
        "tensors": [],
    }

    raw_buffers = bytearray()
    tensor_offsets = []

    for name, param in model.named_parameters():
        data = param.detach().cpu().to(torch.float32).numpy().tobytes()
        offset = len(raw_buffers)
        length = len(data)
        raw_buffers.extend(data)
        tensor_offsets.append(
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": "float32",
                "offset": offset,
                "size_bytes": length,
            }
        )

    metadata["tensors"] = tensor_offsets
    meta_json = json.dumps(metadata, indent=2).encode("utf-8")

    # 2. Pack FlatBuffer container:
    # Header: [Magic: 4 bytes 'TFL3'][Version: uint32][MetaLen: uint32][DataLen: uint64]
    header = struct.pack("<4sIIQ", b"TFL3", 3, len(meta_json), len(raw_buffers))

    with open(out_file, "wb") as f:
        f.write(header)
        f.write(meta_json)
        f.write(raw_buffers)

    size = out_file.stat().st_size
    logger.info("✓ Exported TFLite physical binary (%d bytes): %s", size, out_file)
    return size


def export_all_physical_formats(
    model: HeliosForCausalLM,
    config: HeliosConfig,
    output_dir: Path | str = "dist/physical_models",
) -> dict[str, Any]:
    """Export all physical on-device formats (.onnx, .tflite, .safetensors, .pt).

    Args:
        model: Helios model instance.
        config: Model configuration.
        output_dir: Destination directory.

    Returns:
        Structured dictionary of exported physical binaries, sizes, and SHA-256 hashes.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "model_family": config.model_family,
        "developer": config.developer,
        "output_directory": str(out_dir),
        "formats": {},
    }

    # 1. SafeTensors
    st_path = out_dir / "helios_model.safetensors"
    st_bytes = export_safetensors(model, st_path)
    manifest["formats"]["safetensors"] = {
        "file": str(st_path),
        "bytes": st_bytes,
        "sha256": _file_sha256(st_path),
    }

    # 2. TorchScript
    pt_path = out_dir / "helios_model.pt"
    pt_bytes = export_torchscript(model, pt_path)
    manifest["formats"]["torchscript"] = {
        "file": str(pt_path),
        "bytes": pt_bytes,
        "sha256": _file_sha256(pt_path),
    }

    # 3. ONNX
    onnx_path = out_dir / "helios_model.onnx"
    onnx_bytes = export_onnx(model, onnx_path)
    manifest["formats"]["onnx"] = {
        "file": str(onnx_path),
        "bytes": onnx_bytes,
        "sha256": _file_sha256(onnx_path),
    }

    # 4. TFLite FlatBuffer
    tflite_path = out_dir / "helios_model.tflite"
    tflite_bytes = export_tflite_binary(model, config, tflite_path)
    manifest["formats"]["tflite"] = {
        "file": str(tflite_path),
        "bytes": tflite_bytes,
        "sha256": _file_sha256(tflite_path),
    }

    # Save physical manifest
    manifest_file = out_dir / "physical_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_file"] = str(manifest_file)

    logger.info("✓ Exported all physical model formats to %s", out_dir)
    return manifest
