"""Speculative Decoding Engine for Helios Foundation Models (Feature 46).

Implements Leviathan rejection sampling pairing Nano-20M (draft) with
Flagship-6B (target) with autoregressive K-step lookahead candidate generation,
batched parallel verification, synchronized KV-cache rollbacks, and bonus token emission.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from helios.model.attention import KVCache
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM


@dataclass
class SpeculativeConfig:
    """Configuration for Speculative Decoding Engine."""

    lookahead_k: int = 4
    lookahead_gamma: int | None = None
    temperature: float = 0.0
    top_k: int | None = None
    max_new_tokens: int = 50
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.lookahead_gamma is not None:
            self.lookahead_k = self.lookahead_gamma
        else:
            self.lookahead_gamma = self.lookahead_k
        if self.lookahead_k <= 0:
            raise ValueError(f"lookahead_k must be > 0, got {self.lookahead_k}")
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0.0, got {self.temperature}")


@dataclass
class SpeculativeMetrics:
    """Performance telemetry and acceptance metrics from speculative decoding."""

    total_generated_tokens: int
    total_draft_tokens: int
    accepted_draft_tokens: int
    acceptance_rate: float
    speculative_steps: int
    wall_clock_time_sec: float
    tokens_per_second: float


@dataclass
class SpeculativeGenerationOutput:
    """Output container for speculative decoding containing tokens and telemetry."""

    tokens: torch.Tensor
    metrics: SpeculativeMetrics


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> tuple[int, torch.Tensor]:
    """Sample token index and normalized distribution from logits."""
    if temperature <= 0.0:
        token = int(torch.argmax(logits, dim=-1).item())
        prob = F.softmax(logits, dim=-1)
        return token, prob

    scaled = logits / temperature
    if top_k is not None and top_k > 0 and top_k < scaled.size(-1):
        v, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        scaled[scaled < v[..., -1, None]] = float("-inf")

    prob = F.softmax(scaled, dim=-1)
    token = int(torch.multinomial(prob, num_samples=1).item())
    return token, prob


def _sample_recovery(
    p_dist: torch.Tensor,
    q_dist: torch.Tensor,
    temperature: float = 0.0,
    top_k: int | None = None,
) -> int:
    """Sample recovery token from adjusted distribution max(0, p - q)."""
    if temperature <= 0.0:
        return int(torch.argmax(p_dist, dim=-1).item())

    p_p = torch.clamp(p_dist - q_dist, min=0.0)
    total = p_p.sum()
    if total > 0.0:
        p_p = p_p / total
    else:
        p_p = p_dist

    if top_k is not None and top_k > 0 and top_k < p_p.size(-1):
        v, _ = torch.topk(p_p, min(top_k, p_p.size(-1)))
        p_p[p_p < v[..., -1, None]] = 0.0
        p_p = p_p / (p_p.sum() + 1e-12)

    return int(torch.multinomial(p_p, num_samples=1).item())


class SpeculativeDecodingEngine:
    """Speculative Decoding Engine pairing a fast draft model with a large target model."""

    def __init__(
        self,
        target_model: HeliosForCausalLM | None = None,
        draft_model: HeliosForCausalLM | None = None,
        config: SpeculativeConfig | None = None,
        tokenizer: Any = None,
        lookahead_gamma: int | None = None,
        **kwargs: Any,
    ) -> None:
        if target_model is None and "draft_model" in kwargs:
            draft_model = kwargs.pop("draft_model")
        if target_model is None and "target_model" in kwargs:
            target_model = kwargs.pop("target_model")

        if target_model is not None and draft_model is not None:
            if hasattr(target_model, "config") and hasattr(draft_model, "config"):
                if getattr(target_model.config, "n_layers", 0) < getattr(
                    draft_model.config, "n_layers", 0
                ):
                    target_model, draft_model = draft_model, target_model

        if target_model is None or draft_model is None:
            raise ValueError("Both target_model and draft_model must be provided.")

        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer

        if lookahead_gamma is not None:
            config = config or SpeculativeConfig()
            config.lookahead_k = config.lookahead_gamma = lookahead_gamma
        self.config = config or SpeculativeConfig()
        self.lookahead_gamma = self.config.lookahead_k

        if target_model.config.vocab_size != draft_model.config.vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: target ({target_model.config.vocab_size}) "
                f"!= draft ({draft_model.config.vocab_size})"
            )

    @classmethod
    def from_presets(
        cls,
        target_preset: str = "flagship_6b",
        draft_preset: str = "nano_20m",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        **config_overrides: Any,
    ) -> SpeculativeDecodingEngine:
        """Instantiate speculative engine from preset model configurations."""
        fn_target = getattr(HeliosConfig, target_preset.replace("-", "_"))
        fn_draft = getattr(HeliosConfig, draft_preset.replace("-", "_"))

        cfg_target = fn_target()
        cfg_draft = fn_draft()

        target_model = HeliosForCausalLM(cfg_target).to(device=device, dtype=dtype)
        draft_model = HeliosForCausalLM(cfg_draft).to(device=device, dtype=dtype)

        spec_cfg = SpeculativeConfig(device=device, dtype=dtype, **config_overrides)
        return cls(target_model=target_model, draft_model=draft_model, config=spec_cfg)

    def _init_caches(
        self, batch_size: int, total_max_len: int, device: torch.device | str, dtype: torch.dtype
    ) -> tuple[list[KVCache], list[KVCache]]:
        """Initialize independent KV-caches for target and draft models."""
        target_head_dim = self.target_model.config.d_model // self.target_model.config.n_heads
        draft_head_dim = self.draft_model.config.d_model // self.draft_model.config.n_heads

        target_caches = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=total_max_len,
                n_kv_heads=self.target_model.config.n_kv_heads,
                head_dim=target_head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.target_model.config.n_layers)
        ]
        draft_caches = [
            KVCache(
                max_batch_size=batch_size,
                max_seq_len=total_max_len,
                n_kv_heads=self.draft_model.config.n_kv_heads,
                head_dim=draft_head_dim,
                dtype=dtype,
                device=device,
            )
            for _ in range(self.draft_model.config.n_layers)
        ]
        return target_caches, draft_caches

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        prompt_ids: torch.Tensor | None = None,
    ) -> SpeculativeGenerationOutput:
        """Autoregressively generate tokens using speculative lookahead and verification."""
        t_start = time.perf_counter()

        k = self.config.lookahead_k
        temp = self.config.temperature if temperature is None else temperature
        k_top = self.config.top_k if top_k is None else top_k
        max_tokens = self.config.max_new_tokens if max_new_tokens is None else max_new_tokens
        eos_id = self.config.eos_token_id if eos_token_id is None else eos_token_id

        input_tokens = prompt_tokens if prompt_tokens is not None else prompt_ids
        if input_tokens is None:
            raise ValueError("Either prompt_tokens or prompt_ids must be provided.")

        if input_tokens.ndim == 1:
            input_tokens = input_tokens.unsqueeze(0)

        device = input_tokens.device
        dtype = self.config.dtype
        _, prompt_len = input_tokens.shape
        total_budget = prompt_len + max_tokens + k + 16

        target_caches, draft_caches = self._init_caches(1, total_budget, device, dtype)

        # 1. Warm prefill on both models
        target_out = self.target_model(
            input_ids=input_tokens,
            start_pos=0,
            kv_caches=target_caches,
            use_cache=True,
        )
        self.draft_model(
            input_ids=input_tokens,
            start_pos=0,
            kv_caches=draft_caches,
            use_cache=True,
        )

        generated_ids: list[int] = input_tokens[0].tolist()
        cur_pos = prompt_len
        target_last_logits = target_out["logits"][0, -1, :]

        total_draft_tokens = 0
        accepted_draft_tokens = 0
        speculative_steps = 0

        while len(generated_ids) - prompt_len < max_tokens:
            speculative_steps += 1
            step_budget = max_tokens - (len(generated_ids) - prompt_len)
            curr_k = min(k, step_budget)
            if curr_k <= 0:
                break

            # 2. Draft autoregressive candidate generation (K tokens)
            cand_tokens: list[int] = []
            draft_probs: list[torch.Tensor] = []

            # First draft candidate sampled from target's previous distribution
            first_cand, p0 = _sample_from_logits(target_last_logits, temp, k_top)
            cand_tokens.append(first_cand)
            _, q0 = _sample_from_logits(target_last_logits, temp, k_top)
            draft_probs.append(q0)

            # Lookahead step 2..curr_k
            for j in range(1, curr_k):
                d_in = torch.tensor([[cand_tokens[-1]]], device=device)
                d_out = self.draft_model(
                    input_ids=d_in,
                    start_pos=cur_pos + j - 1,
                    kv_caches=draft_caches,
                    use_cache=True,
                )
                cand_tok, q_dist = _sample_from_logits(d_out["logits"][0, -1, :], temp, k_top)
                cand_tokens.append(cand_tok)
                draft_probs.append(q_dist)

            total_draft_tokens += curr_k

            # 3. Target batched verification forward pass on candidate slice
            cand_tensor = torch.tensor([cand_tokens], device=device)
            target_cand_out = self.target_model(
                input_ids=cand_tensor,
                start_pos=cur_pos,
                kv_caches=target_caches,
                use_cache=True,
            )
            target_cand_logits = target_cand_out["logits"][0]  # (curr_k, vocab_size)

            target_probs: list[torch.Tensor] = [p0]
            for j in range(curr_k - 1):
                _, pj = _sample_from_logits(target_cand_logits[j, :], temp, k_top)
                target_probs.append(pj)

            # 4. Leviathan rejection sampling loop
            rejection_idx: int | None = None
            recovery_token: int | None = None
            eos_accepted = False

            for i in range(curr_k):
                token_i = cand_tokens[i]
                p_i = target_probs[i]
                q_i = draft_probs[i]

                if temp <= 0.0:
                    # Deterministic greedy check
                    target_arg = int(torch.argmax(p_i).item())
                    if token_i == target_arg:
                        accepted_draft_tokens += 1
                        generated_ids.append(token_i)
                        if eos_id is not None and token_i == eos_id:
                            eos_accepted = True
                            eos_pos = cur_pos + i + 1
                            if i == curr_k - 1:
                                eos_t = torch.tensor([[token_i]], device=device)
                                self.draft_model(
                                    input_ids=eos_t,
                                    start_pos=cur_pos + i,
                                    kv_caches=draft_caches,
                                    use_cache=True,
                                )
                            for cache in target_caches:
                                cache.cur_len = eos_pos
                            for cache in draft_caches:
                                cache.cur_len = eos_pos
                            cur_pos = eos_pos
                            break
                    else:
                        rejection_idx = i
                        recovery_token = target_arg
                        break
                else:
                    # Stochastic Leviathan acceptance test
                    p_val = float(p_i[token_i].item())
                    q_val = float(q_i[token_i].item())
                    u = float(torch.rand(1).item())
                    ratio = p_val / max(q_val, 1e-12)

                    if u < min(1.0, ratio):
                        accepted_draft_tokens += 1
                        generated_ids.append(token_i)
                        if eos_id is not None and token_i == eos_id:
                            eos_accepted = True
                            eos_pos = cur_pos + i + 1
                            if i == curr_k - 1:
                                eos_t = torch.tensor([[token_i]], device=device)
                                self.draft_model(
                                    input_ids=eos_t,
                                    start_pos=cur_pos + i,
                                    kv_caches=draft_caches,
                                    use_cache=True,
                                )
                            for cache in target_caches:
                                cache.cur_len = eos_pos
                            for cache in draft_caches:
                                cache.cur_len = eos_pos
                            cur_pos = eos_pos
                            break
                    else:
                        rejection_idx = i
                        recovery_token = _sample_recovery(p_i, q_i, temp, k_top)
                        break

            if eos_accepted:
                break

            # 5. KV-Cache Rollback & Recovery
            if rejection_idx is not None:
                rollback_pos = cur_pos + rejection_idx
                for cache in target_caches:
                    cache.cur_len = rollback_pos
                for cache in draft_caches:
                    cache.cur_len = rollback_pos

                if recovery_token is not None:
                    generated_ids.append(recovery_token)
                    rec_t = torch.tensor([[recovery_token]], device=device)
                    t_rec_out = self.target_model(
                        input_ids=rec_t,
                        start_pos=rollback_pos,
                        kv_caches=target_caches,
                        use_cache=True,
                    )
                    self.draft_model(
                        input_ids=rec_t,
                        start_pos=rollback_pos,
                        kv_caches=draft_caches,
                        use_cache=True,
                    )
                    target_last_logits = t_rec_out["logits"][0, -1, :]
                    cur_pos = rollback_pos + 1
                else:
                    cur_pos = rollback_pos

                if eos_id is not None and recovery_token is not None and recovery_token == eos_id:
                    break
            else:
                # All curr_k candidates accepted: emit bonus token!
                bonus_token, _ = _sample_from_logits(target_cand_logits[-1, :], temp, k_top)
                generated_ids.append(bonus_token)

                bonus_t = torch.tensor([[bonus_token]], device=device)
                t_b_out = self.target_model(
                    input_ids=bonus_t,
                    start_pos=cur_pos + curr_k,
                    kv_caches=target_caches,
                    use_cache=True,
                )
                sync_tokens = torch.tensor([[cand_tokens[-1], bonus_token]], device=device)
                self.draft_model(
                    input_ids=sync_tokens,
                    start_pos=cur_pos + curr_k - 1,
                    kv_caches=draft_caches,
                    use_cache=True,
                )
                target_last_logits = t_b_out["logits"][0, -1, :]
                cur_pos = cur_pos + curr_k + 1

                if eos_id is not None and bonus_token == eos_id:
                    break

        if len(generated_ids) - prompt_len > max_tokens:
            generated_ids = generated_ids[: prompt_len + max_tokens]
        total_gen = len(generated_ids) - prompt_len
        t_elapsed = max(time.perf_counter() - t_start, 1e-6)
        acceptance_rate = (
            (accepted_draft_tokens / total_draft_tokens) if total_draft_tokens > 0 else 0.0
        )
        tok_per_sec = total_gen / t_elapsed

        metrics = SpeculativeMetrics(
            total_generated_tokens=total_gen,
            total_draft_tokens=total_draft_tokens,
            accepted_draft_tokens=accepted_draft_tokens,
            acceptance_rate=acceptance_rate,
            speculative_steps=speculative_steps,
            wall_clock_time_sec=t_elapsed,
            tokens_per_second=tok_per_sec,
        )

        return SpeculativeGenerationOutput(
            tokens=torch.tensor(generated_ids, device=device).unsqueeze(0),
            metrics=metrics,
        )

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_ids: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        prompt_tokens: torch.Tensor | None = None,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Iterator[int]:
        """Stream generated token IDs with speculative speedup."""
        input_tokens = prompt_ids if prompt_ids is not None else prompt_tokens
        if input_tokens is None:
            raise ValueError("Either prompt_ids or prompt_tokens must be provided.")
        out = self.generate(
            prompt_tokens=input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
        )
        prompt_len = input_tokens.shape[1] if input_tokens.ndim > 1 else input_tokens.shape[0]
        for tok in out.tokens[0, prompt_len:].tolist():
            yield int(tok)
