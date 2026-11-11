#!/usr/bin/env python3
"""Clarify the real parameter counts and full-vs-slice status of all six tiers.

The important distinction is:

- **Unique parameters** are the advertised model size. Tied embeddings are
  counted once.
- **State-dict parameters** double-count the tied LM head because
  ``model.state_dict()`` stores ``embed_tokens.weight`` and ``lm_head.weight``
  as two tensors that alias the same storage.

A checkpoint can therefore look "larger" than its advertised size without
being wrong. The honest gate is the unique count plus the expected tensor
count (``9 * n_layers + 3``).

Usage:
    python scripts/audit_full_family_parameters.py
    python scripts/audit_full_family_parameters.py --json-out reports/full_family_parameters.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from helios.model.config import HeliosConfig
from scripts.continue_full_family_lineage import (
    CHECKPOINTS,
    LINEAGE_ROOT,
    TIERS,
    expected_tensor_count,
    training_memory_gib,
    unique_parameter_count,
)

V040_ROOT = CHECKPOINTS / "v040_lineage"


@dataclass
class TierParameterAudit:
    """Parameter and integrity facts for one tier."""

    tier: str
    advertised_unique_parameters: int = 0
    stored_state_parameters: int = 0
    tensor_count: int = 0
    expected_full_tensors: int = 0
    full_model: bool = False
    mode: str = "missing"
    effective_step: int = 0
    parent_path: str = ""
    locally_trainable: bool = False
    estimated_training_gib: float = 0.0
    notes: list[str] = field(default_factory=list)


def load_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load a state dict and its payload metadata."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        return {}, {}
    return state, payload


def audit_checkpoint(tier: str, path: Path, expected: int) -> tuple[int, int, bool, int, str]:
    """Return (state params, tensor count, full model, step, mode) for a checkpoint."""
    if not path.is_dir():
        return 0, 0, False, 0, "missing"
    state_path = path / "state.pt"
    if not state_path.is_file():
        return 0, 0, False, 0, "missing state.pt"
    state, payload = load_state(state_path)
    if not state:
        return 0, 0, False, 0, "empty state"
    tensors = len(state)
    params = sum(tensor.numel() for tensor in state.values())
    is_block = any(key.startswith("block.") for key in state)
    is_full = not is_block and tensors == expected
    mode = "block_slice" if is_block else "full" if is_full else "partial"
    step = int(payload.get("step", 0) or 0)
    return params, tensors, is_full, step, mode


def audit_tier(tier: str) -> TierParameterAudit:
    """Audit the parent, v0.4.0 lineage, and v0.5.0 full lineage for one tier."""
    config = HeliosConfig.from_preset(TIERS[tier]["preset"])
    expected = expected_tensor_count(config)
    result = TierParameterAudit(
        tier=tier,
        advertised_unique_parameters=unique_parameter_count(config),
        expected_full_tensors=expected,
        estimated_training_gib=training_memory_gib(config),
    )

    parent_path = CHECKPOINTS / f"helios_{tier}"
    result.parent_path = str(parent_path)
    if parent_path.is_dir():
        params, tensors, is_full, step, mode = audit_checkpoint(tier, parent_path, expected)
        result.stored_state_parameters = params
        result.tensor_count = tensors
        result.full_model = is_full
        result.mode = mode
        result.effective_step = step
        if is_full:
            result.locally_trainable = training_memory_gib(config) < 8.0
        else:
            result.notes.append("parent is not a full model")
    else:
        result.mode = "missing"
        result.notes.append("no parent checkpoint exists on the drive")

    v040 = V040_ROOT / f"{tier}_continuous" / "latest"
    if v040.is_dir():
        params, tensors, is_full, step, mode = audit_checkpoint(tier, v040, expected)
        result.notes.append(
            f"v0.4.0 lineage: {mode}, {tensors} tensors, {params / 1e6:.2f}M stored params"
        )

    v050 = LINEAGE_ROOT / f"{tier}_full" / "latest"
    if v050.is_dir():
        params, tensors, is_full, step, mode = audit_checkpoint(tier, v050, expected)
        result.notes.append(
            f"v0.5.0 lineage: {mode}, {tensors} tensors, {params / 1e6:.2f}M stored params, "
            f"step {step}"
        )
        if is_full:
            result.stored_state_parameters = params
            result.tensor_count = tensors
            result.full_model = True
            result.mode = "full"
            result.effective_step = step
            result.locally_trainable = training_memory_gib(config) < 8.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    audits = [audit_tier(tier) for tier in TIERS]
    print("Helios six-tier parameter truth table")
    print("=" * 96)
    print(
        "Tier       Advertised unique   Stored state   Tensors  Full?  Locally trainable  Mode"
    )
    for item in audits:
        print(
            f"{item.tier:<10} {item.advertised_unique_parameters / 1e6:>14.2f}M "
            f"{item.stored_state_parameters / 1e6:>13.2f}M "
            f"{item.tensor_count:>7}/{item.expected_full_tensors:<4} "
            f"{'yes' if item.full_model else 'no':>5}  "
            f"{'yes' if item.locally_trainable else 'no':>17}  {item.mode}"
        )
    print("\nNotes")
    for item in audits:
        print(
            f"  {item.tier}: unique={item.advertised_unique_parameters:,} | "
            f"estimated full-training memory={item.estimated_training_gib:.2f} GiB "
            f"(weights+grads+AdamW only)"
        )
        for note in item.notes:
            print(f"    - {note}")

    print("\nDistillation gate")
    flagship = next(item for item in audits if item.tier == "flagship")
    if flagship.full_model:
        print("  Flagship is a full model and can be used as a teacher.")
    else:
        print(
            "  BLOCKED: Flagship is not a full model "
            f"({flagship.tensor_count}/{flagship.expected_full_tensors} tensors). "
            "Distillation cannot start until a full Flagship exists."
        )
    print("  Intended order: Flagship -> Pro -> Core -> Beta -> Alpha -> Nano")
    print("  A staged cascade lets each student become the next smaller teacher.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps([asdict(item) for item in audits], indent=2), encoding="utf-8"
        )
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
