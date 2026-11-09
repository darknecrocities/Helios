"""Checkpoint verification and parameter validation utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass
class CheckpointInfo:
    """Detailed structural metadata for an inspected checkpoint or config."""

    model_name: str
    tier: str
    total_parameters: int
    parameters_millions: float
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    max_seq_len: int
    weight_sha256: str
    format: str


class CheckpointVerifier:
    """Verifies checkpoint SHA-256 digests and integrity before evaluation."""

    @staticmethod
    def compute_sha256(file_path: str | Path) -> str:
        """Compute SHA-256 hash of a file on disk."""
        path = Path(file_path)
        if not path.exists():
            return ""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def verify_parameter_count(
        model: torch.nn.Module, expected_params: int, tolerance: float = 0.05
    ) -> bool:
        """Verify model parameter count matches configuration specification."""
        actual_params = sum(p.numel() for p in model.parameters())
        if expected_params == 0:
            return True
        diff = abs(actual_params - expected_params) / expected_params
        return diff <= tolerance

    def inspect(self, checkpoint_path: str | Path) -> CheckpointInfo:
        """Inspect a model checkpoint file, weights directory, or model YAML config."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

        file_sha256 = self.compute_sha256(path)

        # Case 1: YAML Model Config
        if path.suffix in (".yaml", ".yml"):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            model_data = data.get("model", data)
            arch = model_data.get("architecture", {})
            name = model_data.get("name", path.stem)
            tier = model_data.get("tier", path.stem)

            layers = arch.get("num_layers", 12)
            hidden = arch.get("hidden_dim", 768)
            q_heads = arch.get("num_attention_heads", 12)
            kv_heads = arch.get("num_key_value_heads", 4)
            seq_len = arch.get("max_seq_len", 8192)

            params = model_data.get("parameters_count", 0)
            if not params:
                params = int(layers * (12 * (hidden**2)) * 1.1)

            return CheckpointInfo(
                model_name=name,
                tier=tier,
                total_parameters=params,
                parameters_millions=params / 1e6,
                num_layers=layers,
                hidden_dim=hidden,
                num_attention_heads=q_heads,
                num_key_value_heads=kv_heads,
                max_seq_len=seq_len,
                weight_sha256=file_sha256,
                format="yaml_spec",
            )

        # Case 2: PyTorch checkpoint
        if path.is_file():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            total_params = sum(
                v.numel() for v in state_dict.values() if isinstance(v, torch.Tensor)
            )

            return CheckpointInfo(
                model_name=path.stem,
                tier="custom",
                total_parameters=total_params,
                parameters_millions=total_params / 1e6,
                num_layers=sum(1 for k in state_dict.keys() if "layers" in k and "weight" in k),
                hidden_dim=0,
                num_attention_heads=0,
                num_key_value_heads=0,
                max_seq_len=8192,
                weight_sha256=file_sha256,
                format="pytorch_pt",
            )

        raise ValueError(f"Unsupported checkpoint format: {path}")
