"""Helios Parameter Inspector and Model Sizing Analyzer.

Provides exact non-duplicating parameter counting, theoretical calculations,
precision-aware memory footprint estimation, and hierarchical module breakdown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn

if TYPE_CHECKING:
    from helios.model.config import HeliosConfig


@dataclass
class LayerBreakdown:
    """Parameter breakdown for a single transformer layer."""

    layer_index: int
    input_norm_params: int
    q_proj_params: int
    k_proj_params: int
    v_proj_params: int
    o_proj_params: int
    total_attn_params: int
    post_attn_norm_params: int
    gate_proj_params: int
    up_proj_params: int
    down_proj_params: int
    total_ffn_params: int
    total_layer_params: int


@dataclass
class ParameterInspectionReport:
    """Detailed inspection report for a neural network model."""

    total_parameters: int
    trainable_parameters: int
    non_embedding_parameters: int
    embedding_parameters: int
    tied_embeddings: bool
    memory_footprint_mb: dict[str, float]
    layers: list[LayerBreakdown]
    preset_match: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)

    def summary(self) -> str:
        """Generate human-readable markdown summary."""
        total_m = self.total_parameters / 1e6
        non_emb_m = self.non_embedding_parameters / 1e6
        emb_m = self.embedding_parameters / 1e6
        lines = [
            "# Helios Parameter Inspection Report",
            f"- **Total Parameters**: {self.total_parameters:,} ({total_m:.2f}M)",
            f"- **Trainable Parameters**: {self.trainable_parameters:,}",
            f"- **Non-Embedding Parameters**: {self.non_embedding_parameters:,} ({non_emb_m:.2f}M)",
            f"- **Embedding Parameters**: {self.embedding_parameters:,} ({emb_m:.2f}M)",
            f"- **Tied Embeddings**: {self.tied_embeddings}",
            f"- **Preset Match**: {self.preset_match or 'Custom'}",
            "",
            "## Estimated Memory Footprint",
            f"- **FP32**: {self.memory_footprint_mb['fp32']:.2f} MB",
            f"- **FP16 / BF16**: {self.memory_footprint_mb['fp16']:.2f} MB",
            f"- **INT8**: {self.memory_footprint_mb['int8']:.2f} MB",
            f"- **INT4**: {self.memory_footprint_mb['int4']:.2f} MB",
            "",
            f"## Transformer Layers Breakdown ({len(self.layers)} layers)",
        ]
        if self.layers:
            first = self.layers[0]
            lines.append(f"- **Per-Layer Total**: {first.total_layer_params:,} parameters")
            lines.append(
                f"  * Attention: {first.total_attn_params:,} (Q={first.q_proj_params:,}, "
                f"K={first.k_proj_params:,}, V={first.v_proj_params:,}, O={first.o_proj_params:,})"
            )
            lines.append(
                f"  * Feed-Forward (SwiGLU): {first.total_ffn_params:,} "
                f"(Gate={first.gate_proj_params:,}, "
                f"Up={first.up_proj_params:,}, Down={first.down_proj_params:,})"
            )
            lines.append(f"  * Norms: {first.input_norm_params + first.post_attn_norm_params:,}")

        return "\n".join(lines)


def calculate_theoretical_parameters(config: HeliosConfig) -> dict[str, int]:
    """Calculate exact theoretical parameter counts from configuration."""
    v = config.vocab_size
    d = config.d_model
    num_layers = config.n_layers
    h_q = config.n_heads
    h_kv = config.n_kv_heads
    d_ffn = getattr(config, "d_ffn", getattr(config, "d_ff", 768))
    tied = getattr(config, "tie_word_embeddings", getattr(config, "tie_embeddings", True))

    head_dim = d // h_q
    q_params = d * (h_q * head_dim)
    k_params = d * (h_kv * head_dim)
    v_params = d * (h_kv * head_dim)
    o_params = (h_q * head_dim) * d
    attn_params = q_params + k_params + v_params + o_params

    ffn_params = 3 * d * d_ffn
    norms_per_layer = 2 * d
    layer_total = attn_params + ffn_params + norms_per_layer
    layers_total = num_layers * layer_total

    final_norm_params = d
    embed_params = v * d
    lm_head_params = 0 if tied else (v * d)

    non_embed_params = layers_total + final_norm_params
    total_params = embed_params + non_embed_params + lm_head_params

    return {
        "total_parameters": total_params,
        "non_embedding_parameters": non_embed_params,
        "embedding_parameters": embed_params,
        "layer_total": layer_total,
        "attn_params": attn_params,
        "ffn_params": ffn_params,
    }


def inspect_parameters(model: nn.Module) -> ParameterInspectionReport:
    """Inspect a Helios causal language model and compute exact parameter metrics.

    Deduplicates tied parameter buffers (such as lm_head.weight == embed_tokens.weight)
    using unique data pointers to prevent double-counting.
    """
    seen_pointers: set[int] = set()
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        ptr = p.data_ptr()
        if ptr not in seen_pointers:
            seen_pointers.add(ptr)
            numel = p.numel()
            total_params += numel
            if p.requires_grad:
                trainable_params += numel

    # Detect tied embeddings
    embed_tokens = getattr(model, "embed_tokens", None)
    lm_head = getattr(model, "lm_head", None)
    tied = False
    embedding_params = 0

    if embed_tokens is not None and lm_head is not None:
        tied = embed_tokens.weight.data_ptr() == lm_head.weight.data_ptr()
        embedding_params = embed_tokens.weight.numel()
        if not tied:
            embedding_params += lm_head.weight.numel()
    elif embed_tokens is not None:
        embedding_params = embed_tokens.weight.numel()

    non_embedding_params = total_params - embedding_params

    # Memory footprint in MiB
    mb_denom = 1024.0 * 1024.0
    memory_footprint = {
        "fp32": (total_params * 4.0) / mb_denom,
        "fp16": (total_params * 2.0) / mb_denom,
        "bf16": (total_params * 2.0) / mb_denom,
        "int8": (total_params * 1.0) / mb_denom,
        "int4": (total_params * 0.5) / mb_denom,
    }

    # Per-layer breakdown
    layers_breakdown: list[LayerBreakdown] = []
    layers = getattr(model, "layers", None)
    if layers is not None and isinstance(layers, nn.ModuleList):
        for idx, block in enumerate(layers):
            attn = getattr(block, "attention", None)
            ffn = getattr(block, "feed_forward", None)
            in_norm = getattr(block, "input_layernorm", None)
            post_norm = getattr(block, "post_attention_layernorm", None)

            q_p = attn.q_proj.weight.numel() if attn and hasattr(attn, "q_proj") else 0
            k_p = attn.k_proj.weight.numel() if attn and hasattr(attn, "k_proj") else 0
            v_p = attn.v_proj.weight.numel() if attn and hasattr(attn, "v_proj") else 0
            o_p = attn.o_proj.weight.numel() if attn and hasattr(attn, "o_proj") else 0
            attn_tot = q_p + k_p + v_p + o_p

            g_p = ffn.gate_proj.weight.numel() if ffn and hasattr(ffn, "gate_proj") else 0
            u_p = ffn.up_proj.weight.numel() if ffn and hasattr(ffn, "up_proj") else 0
            d_p = ffn.down_proj.weight.numel() if ffn and hasattr(ffn, "down_proj") else 0
            ffn_tot = g_p + u_p + d_p

            in_n = in_norm.weight.numel() if in_norm and hasattr(in_norm, "weight") else 0
            post_n = post_norm.weight.numel() if post_norm and hasattr(post_norm, "weight") else 0

            tot_l = in_n + attn_tot + post_n + ffn_tot
            layers_breakdown.append(
                LayerBreakdown(
                    layer_index=idx,
                    input_norm_params=in_n,
                    q_proj_params=q_p,
                    k_proj_params=k_p,
                    v_proj_params=v_p,
                    o_proj_params=o_p,
                    total_attn_params=attn_tot,
                    post_attn_norm_params=post_n,
                    gate_proj_params=g_p,
                    up_proj_params=u_p,
                    down_proj_params=d_p,
                    total_ffn_params=ffn_tot,
                    total_layer_params=tot_l,
                )
            )

    # Preset matching logic
    preset_match = None
    if 14_000_000 <= total_params <= 15_500_000:
        preset_match = "nano_15m"
    elif 24_000_000 <= total_params <= 26_000_000:
        preset_match = "nano_25m"
    elif 37_000_000 <= total_params <= 40_000_000:
        preset_match = "nano_35m"

    return ParameterInspectionReport(
        total_parameters=total_params,
        trainable_parameters=trainable_params,
        non_embedding_parameters=non_embedding_params,
        embedding_parameters=embedding_params,
        tied_embeddings=tied,
        memory_footprint_mb=memory_footprint,
        layers=layers_breakdown,
        preset_match=preset_match,
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Simple parameter counting function compliant with PROJECT.md contract.

    Returns:
        Dict with keys: 'total_parameters' and 'trainable_parameters'.
    """
    report = inspect_parameters(model)
    return {
        "total_parameters": report.total_parameters,
        "trainable_parameters": report.trainable_parameters,
    }
