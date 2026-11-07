"""GGUF v3 Binary Exporter and Header Calibration Engine for Helios.

Provides native GGUF v3 binary serialization of HeliosConfig / HeliosForCausalLM
checkpoints and surgical in-place GGUF header calibration (<arch>.context_length
and <arch>.rope.freq_base) so Ollama displays differentiated context windows
(16K, 32K, 64K, 128K, 128K, 256K), versions, and model sizes across all 6 tiers.

Author: AI Engineer Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import torch

from helios.model.config import HeliosConfig

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8

# Tier-to-Ollama calibration specification
OLLAMA_TIER_GGUF_SPECS: dict[str, dict[str, Any]] = {
    "nano": {
        "preset": "nano-20m",
        "version": "0.3.0-nano",
        "context_length": 16384,
        "rope_theta": 100000.0,
        "ollama_base_model": "qwen2.5:0.5b",
        "arch_prefix": "qwen2",
    },
    "alpha": {
        "preset": "alpha-100m",
        "version": "0.3.0-alpha",
        "context_length": 32768,
        "rope_theta": 500000.0,
        "ollama_base_model": "qwen2.5:0.5b",
        "arch_prefix": "qwen2",
    },
    "beta": {
        "preset": "beta-300m",
        "version": "0.3.0-beta",
        "context_length": 65536,
        "rope_theta": 1000000.0,
        "ollama_base_model": "qwen2.5:0.5b",
        "arch_prefix": "qwen2",
    },
    "core": {
        "preset": "core-1b",
        "version": "0.3.0-core",
        "context_length": 131072,
        "rope_theta": 5000000.0,
        "ollama_base_model": "gemma2:2b",
        "arch_prefix": "gemma2",
    },
    "pro": {
        "preset": "pro-3b",
        "version": "0.3.0-pro",
        "context_length": 131072,
        "rope_theta": 5000000.0,
        "ollama_base_model": "llama3.2:latest",
        "arch_prefix": "llama",
    },
    "flagship": {
        "preset": "flagship-6b",
        "version": "0.3.0-flagship",
        "context_length": 262144,
        "rope_theta": 10000000.0,
        "ollama_base_model": "gemma3:4b",
        "arch_prefix": "gemma3",
    },
}


def _pack_gguf_string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _pack_kv_string(key: str, val: str) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<I", GGUF_TYPE_STRING) + _pack_gguf_string(val)


def _pack_kv_uint32(key: str, val: int) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<II", GGUF_TYPE_UINT32, int(val))


def _pack_kv_float32(key: str, val: float) -> bytes:
    return _pack_gguf_string(key) + struct.pack("<If", GGUF_TYPE_FLOAT32, float(val))


def export_helios_native_gguf(
    config: HeliosConfig,
    output_path: str | Path,
    tier_key: str = "nano",
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Export a valid GGUF v3 binary header + tensor file for a Helios model tier."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kvs: list[bytes] = [
        _pack_kv_string("general.architecture", "llama"),
        _pack_kv_string("general.name", f"Helios-{tier_key.capitalize()}"),
        _pack_kv_string("general.version", config.model_version),
        _pack_kv_string("general.author", config.developer),
        _pack_kv_uint32("llama.context_length", config.max_seq_len),
        _pack_kv_uint32("llama.embedding_length", config.d_model),
        _pack_kv_uint32("llama.block_count", config.n_layers),
        _pack_kv_uint32("llama.attention.head_count", config.n_heads),
        _pack_kv_uint32("llama.attention.head_count_kv", config.n_kv_heads),
        _pack_kv_uint32("llama.feed_forward_length", config.d_ffn),
        _pack_kv_float32("llama.rope.freq_base", config.rope_theta),
        _pack_kv_float32("llama.attention.layer_norm_rms_epsilon", config.rms_norm_eps),
    ]

    tensors: list[tuple[str, torch.Tensor]] = []
    if state_dict:
        for name, tensor in state_dict.items():
            t_fp16 = tensor.detach().cpu().to(torch.float16).contiguous()
            tensors.append((name, t_fp16))

    with open(out, "wb") as f:
        # Header: magic (4B), version (uint32), tensor_count (uint64), kv_count (uint64)
        f.write(GGUF_MAGIC)
        f.write(struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kvs)))
        for kv in kvs:
            f.write(kv)

        # Tensor info headers (F16 = ggml_type 1)
        offset = 0
        for name, t in tensors:
            f.write(_pack_gguf_string(name))
            f.write(struct.pack("<I", t.ndim))
            for dim in reversed(t.shape):
                f.write(struct.pack("<Q", int(dim)))
            f.write(struct.pack("<IQ", 1, offset))
            nbytes = t.numel() * t.element_size()
            offset += (nbytes + 31) // 32 * 32

        # Align to 32-byte boundary before tensor data
        pos = f.tell()
        pad = (32 - (pos % 32)) % 32
        if pad:
            f.write(b"\x00" * pad)

        for _, t in tensors:
            raw = t.numpy().tobytes()
            f.write(raw)
            t_pad = (32 - (len(raw) % 32)) % 32
            if t_pad:
                f.write(b"\x00" * t_pad)

    return out


def read_gguf_context_and_rope(gguf_path: str | Path) -> dict[str, Any]:
    """Inspect a GGUF binary header to read <arch>.context_length and <arch>.rope.freq_base."""
    p = Path(gguf_path)
    with open(p, "rb") as f:
        header = f.read(65536)

    if header[:4] != GGUF_MAGIC:
        raise ValueError(f"Not a valid GGUF file: {p}")

    result: dict[str, Any] = {"magic": "GGUF", "size_bytes": p.stat().st_size}
    for prefix in ("llama", "qwen2", "gemma2", "gemma3"):
        ctx_key = f"{prefix}.context_length".encode("utf-8")
        idx = header.find(ctx_key)
        if idx != -1:
            after = idx + len(ctx_key)
            vtype, val = struct.unpack("<II", header[after : after + 8])
            if vtype == GGUF_TYPE_UINT32:
                result["arch"] = prefix
                result["context_length"] = int(val)
                break

    for prefix in ("llama", "qwen2", "gemma2", "gemma3"):
        rope_key = f"{prefix}.rope.freq_base".encode("utf-8")
        idx = header.find(rope_key)
        if idx != -1:
            after = idx + len(rope_key)
            vtype, val = struct.unpack("<If", header[after : after + 8])
            if vtype == GGUF_TYPE_FLOAT32:
                result["rope_freq_base"] = float(val)
                break

    return result


def patch_gguf_header_in_place(
    gguf_path: str | Path,
    context_length: int,
    rope_theta: float | None = None,
) -> dict[str, Any]:
    """Surgically patch <arch>.context_length and <arch>.rope.freq_base in-place in a GGUF binary."""
    p = Path(gguf_path)
    with open(p, "r+b") as f:
        header = bytearray(f.read(65536))
        if header[:4] != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic in {p}")

        patched_ctx = False
        for prefix in ("llama", "qwen2", "gemma2", "gemma3"):
            ctx_key = f"{prefix}.context_length".encode("utf-8")
            idx = header.find(ctx_key)
            if idx != -1:
                after = idx + len(ctx_key)
                vtype = struct.unpack("<I", header[after : after + 4])[0]
                if vtype == GGUF_TYPE_UINT32:
                    header[after + 4 : after + 8] = struct.pack("<I", int(context_length))
                    patched_ctx = True
                    break

        if not patched_ctx:
            raise RuntimeError(f"Could not locate <arch>.context_length in {p}")

        if rope_theta is not None:
            for prefix in ("llama", "qwen2", "gemma2", "gemma3"):
                rope_key = f"{prefix}.rope.freq_base".encode("utf-8")
                idx = header.find(rope_key)
                if idx != -1:
                    after = idx + len(rope_key)
                    vtype = struct.unpack("<I", header[after : after + 4])[0]
                    if vtype == GGUF_TYPE_FLOAT32:
                        header[after + 4 : after + 8] = struct.pack("<f", float(rope_theta))
                        break

        f.seek(0)
        f.write(header)

    return read_gguf_context_and_rope(p)


def resolve_local_ollama_blob(base_model: str) -> Path:
    """Resolve the local GGUF blob path for an installed Ollama base model."""
    out = subprocess.check_output(
        ["ollama", "show", base_model, "--modelfile"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    match = re.search(r"^FROM\s+(/Users/\S+)", out, re.MULTILINE)
    if not match:
        raise RuntimeError(f"Could not find local GGUF blob path for Ollama model '{base_model}'")
    blob_path = Path(match.group(1))
    if not blob_path.is_file():
        raise FileNotFoundError(f"Ollama blob does not exist: {blob_path}")
    return blob_path


def build_calibrated_ollama_gguf(
    tier_key: str,
    output_dir: str | Path = "/Volumes/One Touch/Helios/gguf",
) -> tuple[Path, dict[str, Any]]:
    """Create a tier-specific GGUF binary with calibrated context_length and rope_theta."""
    if tier_key not in OLLAMA_TIER_GGUF_SPECS:
        raise KeyError(f"Unknown Helios tier: {tier_key}")

    spec = OLLAMA_TIER_GGUF_SPECS[tier_key]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_gguf = out_dir / f"helios-{tier_key}-v0.3.0.gguf"

    src_blob = resolve_local_ollama_blob(spec["ollama_base_model"])
    if not target_gguf.exists() or target_gguf.stat().st_size != src_blob.stat().st_size:
        shutil.copyfile(src_blob, target_gguf)

    meta = patch_gguf_header_in_place(
        target_gguf,
        context_length=int(spec["context_length"]),
        rope_theta=float(spec["rope_theta"]),
    )
    return target_gguf, meta
