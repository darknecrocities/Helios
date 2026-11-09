"""Tier-aware reasoning context optimizer for the Helios model family.

Provides per-tier context/CoT budgets, attention-sink + sliding-window KV cache
compression, dynamic NTK scaling factor selection, and priority-based prompt packing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ContextProfile:
    """Context and reasoning budget for one model tier."""

    tier: str
    max_context: int
    cot_budget: int
    base_train_len: int
    sink_tokens: int = 4


CONTEXT_PROFILES: dict[str, ContextProfile] = {
    "nano": ContextProfile("nano", 16_384, 2_048, 2_048),
    "alpha": ContextProfile("alpha", 32_768, 4_096, 4_096),
    "beta": ContextProfile("beta", 65_536, 8_192, 8_192),
    "core": ContextProfile("core", 131_072, 16_384, 8_192),
    "pro": ContextProfile("pro", 131_072, 24_576, 8_192),
    "flagship": ContextProfile("flagship", 262_144, 32_768, 8_192),
}


def resolve_tier(preset: str) -> str:
    """Map a preset name (e.g. 'nano-20m', 'alpha_100m') to a tier key."""
    low = preset.lower()
    for tier in CONTEXT_PROFILES:
        if tier in low:
            return tier
    return "nano"


class ReasoningContextOptimizer:
    """Packs prompts and compresses KV caches according to tier budgets."""

    def __init__(self, preset: str) -> None:
        self.profile = CONTEXT_PROFILES[resolve_tier(preset)]

    def ntk_scaling_factor(self, seq_len: int) -> float:
        """Return RoPE scaling factor (>=1.0) when seq_len exceeds base training length."""
        base = self.profile.base_train_len
        if seq_len <= base:
            return 1.0
        return min(seq_len, self.profile.max_context) / base

    def compress_kv_cache_with_sinks(
        self, keys: torch.Tensor, values: torch.Tensor, window: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep the first `sink_tokens` positions plus the last `window` positions.

        Tensors are shaped (batch, heads, seq, head_dim).
        """
        seq = keys.shape[-2]
        sinks = self.profile.sink_tokens
        if seq <= sinks + window:
            return keys, values
        k = torch.cat([keys[..., :sinks, :], keys[..., -window:, :]], dim=-2)
        v = torch.cat([values[..., :sinks, :], values[..., -window:, :]], dim=-2)
        return k, v

    def pack_reasoning_context(
        self,
        system: str,
        facts: list[str],
        history: list[str],
        query: str,
        token_len=lambda s: len(s.split()),
    ) -> str:
        """Pack segments by priority: system > query > facts > recent history.

        Reserves the tier CoT budget for generated reasoning.
        """
        budget = self.profile.max_context - self.profile.cot_budget
        used = token_len(system) + token_len(query)
        kept_facts: list[str] = []
        for fact in facts:
            cost = token_len(fact)
            if used + cost > budget:
                break
            kept_facts.append(fact)
            used += cost
        kept_hist: list[str] = []
        for turn in reversed(history):
            cost = token_len(turn)
            if used + cost > budget:
                break
            kept_hist.insert(0, turn)
            used += cost
        return "\n".join([system, *kept_facts, *kept_hist, query])
