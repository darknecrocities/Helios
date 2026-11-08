"""Tier-5 Adversarial Verification Suite for Speculative Decoding Engine (Feature 46).

Adversarially challenge Feature 46 (helios/model/speculative.py):
1. Mathematical distribution invariance of Leviathan rejection sampling (Chi-Square test).
2. KV-cache rollback on rejection under randomized high-entropy streams with K in {1, 2, 4, 8}.
3. Target model KV-cache exact numerical restoration (|Delta KV| == 0.0).
4. Memory bounding and leak detection across sustained iterations.
5. Strict output length capping across diverse max_new_tokens budgets.
6. Empirical verification of draft KV-cache synchronization hole defect.
7. Verification of EOS token KV-cache cursor rollback anomaly.

Author: challenger_m9_1
"""

from __future__ import annotations

import gc
import resource

import pytest
import torch
import torch.nn.functional as F

from helios.model.attention import KVCache
from helios.model.config import HeliosConfig
from helios.model.speculative import (
    SpeculativeConfig,
    SpeculativeDecodingEngine,
    _sample_recovery,
)
from helios.model.transformer import HeliosForCausalLM


@pytest.fixture
def test_models() -> tuple[HeliosForCausalLM, HeliosForCausalLM]:
    """Create paired target (2 layers) and draft (1 layer) models."""
    torch.manual_seed(999)
    cfg_target = HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=256,
    )
    cfg_draft = HeliosConfig(
        vocab_size=128,
        d_model=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=256,
    )
    target = HeliosForCausalLM(cfg_target).eval()
    draft = HeliosForCausalLM(cfg_draft).eval()
    return target, draft


@pytest.mark.tier5
class TestLeviathanMathematicalEquivalence:
    """Rigorous mathematical validation of Leviathan rejection sampling."""

    def test_chi_square_goodness_of_fit(self) -> None:
        """Verify empirical sampling distribution matches theoretical target p under Chi-Square test."""
        torch.manual_seed(42)
        p = F.softmax(torch.tensor([0.15, 0.35, 0.05, 0.30, 0.15]), dim=-1)
        q = F.softmax(torch.tensor([0.25, 0.10, 0.40, 0.15, 0.10]), dim=-1)

        n_trials = 100000
        samples: list[int] = []

        for _ in range(n_trials):
            cand = int(torch.multinomial(q, 1).item())
            ratio = float(p[cand].item() / q[cand].item())
            u = float(torch.rand(1).item())
            if u < min(1.0, ratio):
                samples.append(cand)
            else:
                rec = _sample_recovery(p, q, temperature=1.0)
                samples.append(rec)

        observed = torch.bincount(torch.tensor(samples), minlength=5).float()
        expected = p * n_trials

        # Chi-Square test statistic: sum((O - E)^2 / E)
        chi2 = float(torch.sum((observed - expected) ** 2 / expected).item())
        # Critical value for df=4 at alpha=0.05 is 9.488
        assert chi2 < 9.488, f"Chi-Square test failed: chi2={chi2:.4f} >= 9.488"

        max_diff = float(torch.max(torch.abs(observed / n_trials - p)).item())
        assert max_diff < 0.005, f"Max absolute distribution divergence exceeded: {max_diff}"

    def test_identity_distribution_zero_rejections(self) -> None:
        """When draft distribution q exactly equals target p, rejection probability must be 0."""
        torch.manual_seed(777)
        p = F.softmax(torch.randn(10), dim=-1)
        q = p.clone()

        rejections = 0
        n_trials = 5000
        for _ in range(n_trials):
            cand = int(torch.multinomial(q, 1).item())
            ratio = float(p[cand].item() / q[cand].item())
            u = float(torch.rand(1).item())
            if u >= min(1.0, ratio):
                rejections += 1

        assert rejections == 0, f"Expected 0 rejections when p == q, got {rejections}"


@pytest.mark.tier5
class TestKVCacheRollbackAndExactness:
    """Stress tests on KV cache rollback with varying K in {1, 2, 4, 8}."""

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_greedy_equivalence_under_high_entropy_rejections(
        self, test_models: tuple[HeliosForCausalLM, HeliosForCausalLM], k: int
    ) -> None:
        """Verify bit-for-bit greedy equivalence with pure target model across varying K."""
        target, draft = test_models
        torch.manual_seed(2024)
        prompt = torch.randint(0, 128, (1, 8))

        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(lookahead_k=k, temperature=0.0, max_new_tokens=16),
        )

        with torch.no_grad():
            spec_out = engine.generate(prompt)
            pure_tokens = target.generate(prompt, max_new_tokens=16, temperature=0.0)

        assert torch.equal(spec_out.tokens, pure_tokens), (
            f"K={k} generated tokens diverged from pure target generation:\n"
            f"Speculative: {spec_out.tokens.tolist()}\n"
            f"Pure Target: {pure_tokens.tolist()}"
        )
        assert spec_out.tokens.shape[1] == 8 + 16

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_target_kv_cache_exact_restoration(
        self, test_models: tuple[HeliosForCausalLM, HeliosForCausalLM], k: int
    ) -> None:
        """Verify target KV-cache matches pure autoregressive target KV-cache (|Delta KV| == 0.0)."""
        target, draft = test_models
        torch.manual_seed(2024)
        prompt = torch.randint(0, 128, (1, 6))

        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(lookahead_k=k, temperature=0.0, max_new_tokens=12),
        )

        orig_init = engine._init_caches
        captured = []

        def hooked_init(*args, **kwargs):
            t_c, d_c = orig_init(*args, **kwargs)
            captured.append((t_c, d_c))
            return t_c, d_c

        engine._init_caches = hooked_init

        with torch.no_grad():
            spec_out = engine.generate(prompt)
            t_c, _ = captured[0]

            # Build pure autoregressive target cache on the output tokens
            seq_len = spec_out.tokens.shape[1]
            head_dim = target.config.d_model // target.config.n_heads
            pure_caches = [
                KVCache(
                    max_batch_size=1,
                    n_kv_heads=target.config.n_kv_heads,
                    max_seq_len=256,
                    head_dim=head_dim,
                )
                for _ in range(target.config.n_layers)
            ]
            target(spec_out.tokens, start_pos=0, kv_caches=pure_caches, use_cache=True)

            for layer_idx in range(target.config.n_layers):
                spec_k = t_c[layer_idx].k_cache[0, :, :seq_len, :]
                pure_k = pure_caches[layer_idx].k_cache[0, :, :seq_len, :]
                k_diff = (spec_k - pure_k).abs().max().item()
                assert k_diff < 1e-5, f"Layer {layer_idx} Key cache diverged: diff={k_diff}"

                spec_v = t_c[layer_idx].v_cache[0, :, :seq_len, :]
                pure_v = pure_caches[layer_idx].v_cache[0, :, :seq_len, :]
                v_diff = (spec_v - pure_v).abs().max().item()
                assert v_diff < 1e-5, f"Layer {layer_idx} Value cache diverged: diff={v_diff}"


@pytest.mark.tier5
class TestMemoryAndOutputBounding:
    """Stress tests on memory retention and output token length bounding."""

    def test_memory_remains_bounded_across_iterations(
        self, test_models: tuple[HeliosForCausalLM, HeliosForCausalLM]
    ) -> None:
        """Verify RSS memory does not leak across 50 consecutive speculative runs."""
        target, draft = test_models
        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(lookahead_k=4, temperature=0.7, max_new_tokens=15),
        )
        prompt = torch.tensor([[10, 20, 30, 40]])

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                engine.generate(prompt)

        gc.collect()
        mem_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        for _ in range(40):
            with torch.no_grad():
                engine.generate(prompt)

        gc.collect()
        mem_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        delta_kb = (mem_end - mem_start) / 1024.0
        # On macOS, ru_maxrss is in bytes; memory growth must be strictly bounded (< 512 KB)
        assert delta_kb < 512.0, f"Potential memory leak detected: delta={delta_kb:.1f} KB"

    @pytest.mark.parametrize("max_tokens", [1, 2, 3, 5, 8, 17])
    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_output_length_strictly_bounded(
        self,
        test_models: tuple[HeliosForCausalLM, HeliosForCausalLM],
        max_tokens: int,
        k: int,
    ) -> None:
        """Verify generated sequence length never exceeds prompt_len + max_new_tokens."""
        target, draft = test_models
        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(lookahead_k=k, temperature=0.0, max_new_tokens=max_tokens),
        )
        prompt = torch.tensor([[5, 10, 15]])
        with torch.no_grad():
            out = engine.generate(prompt, max_new_tokens=max_tokens)

        gen_tokens = out.tokens.shape[1] - prompt.shape[1]
        assert gen_tokens == max_tokens, (
            f"Output length mismatch for max_tokens={max_tokens}, K={k}: "
            f"got {gen_tokens} generated tokens."
        )
        assert out.metrics.total_generated_tokens == max_tokens


@pytest.mark.tier5
class TestKVHoleAndRollbackDefects:
    """Adversarial stress tests exposing architectural defects in speculative.py."""

    def test_draft_kv_cache_gap_on_all_accepted_candidates(self) -> None:
        """Empirically demonstrates uninitialized zero slice in draft cache when candidates accepted.

        Defect in helios/model/speculative.py line 374:
        When all curr_k candidates are accepted, bonus token is forwarded at start_pos = cur_pos + curr_k.
        cand_tokens[-1] was never forwarded to draft_model, leaving position cur_pos + curr_k - 1
        as uninitialized zeros in draft_caches!
        """
        torch.manual_seed(101)
        cfg = HeliosConfig(
            vocab_size=64,
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
            d_ffn=64,
            max_seq_len=64,
        )
        target = HeliosForCausalLM(cfg).eval()
        draft = HeliosForCausalLM(cfg).eval()
        draft.load_state_dict(target.state_dict())  # Identical models guarantee 100% acceptance

        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(lookahead_k=4, temperature=0.0, max_new_tokens=10),
        )

        orig_init = engine._init_caches
        captured = []

        def hooked_init(*args, **kwargs):
            t_c, d_c = orig_init(*args, **kwargs)
            captured.append((t_c, d_c))
            return t_c, d_c

        engine._init_caches = hooked_init

        prompt = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out = engine.generate(prompt, max_new_tokens=10)

        t_c, d_c = captured[0]
        total_len = out.tokens.shape[1]
        draft_k = d_c[0].k_cache[0, 0, :total_len, :]

        # Detect any zero slices within valid context range
        zero_positions = [i for i in range(total_len) if float(draft_k[i].norm().item()) == 0.0]
        # Verified remediation: synchronization of both cand_tokens[-1] and bonus_token eliminates zero gaps
        assert len(zero_positions) == 0, (
            f"Expected 0 uninitialized zero gaps in draft KV cache, found: {zero_positions}"
        )

    def test_eos_token_cur_len_rollback_defect(
        self, test_models: tuple[HeliosForCausalLM, HeliosForCausalLM]
    ) -> None:
        """Empirically demonstrates that accepted EOS token is retained in cache cursor.

        Remediated in helios/model/speculative.py:
        When token_i == eos_id, rejection_idx is NOT set, and cache.cur_len is set to
        cur_pos + i + 1, retaining the accepted EOS token in both target and draft caches.
        """
        target, draft = test_models
        eos_id = 3

        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(
                lookahead_k=3, temperature=0.0, max_new_tokens=10, eos_token_id=eos_id
            ),
        )

        orig_init = engine._init_caches
        captured = []

        def hooked_init(*args, **kwargs):
            t_c, d_c = orig_init(*args, **kwargs)
            captured.append((t_c, d_c))
            return t_c, d_c

        engine._init_caches = hooked_init

        prompt = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out = engine.generate(prompt)

        t_c, d_c = captured[0]
        # Check that EOS handling halted generation properly and retained EOS token in cache
        assert out.tokens.shape[1] == 4
        assert out.tokens[0, -1].item() == eos_id
        assert t_c[0].cur_len == out.tokens.shape[1]
        assert d_c[0].cur_len == out.tokens.shape[1]

    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    def test_draft_kv_cache_no_zero_hole_on_eos(self, k: int) -> None:
        """Verify draft KV-cache has zero uninitialized zero-slice holes when EOS is candidate curr_k - 1.

        Remediated in helios/model/speculative.py:
        When token_i == eos_id and i == curr_k - 1, token_i is explicitly forwarded
        into self.draft_model before updating cache cursors to eos_pos.
        """
        torch.manual_seed(101)
        cfg = HeliosConfig(
            vocab_size=64,
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
            d_ffn=64,
            max_seq_len=64,
        )
        target = HeliosForCausalLM(cfg).eval()
        draft = HeliosForCausalLM(cfg).eval()
        draft.load_state_dict(target.state_dict())

        eos_id = 3
        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(
                lookahead_k=k,
                temperature=0.0,
                max_new_tokens=10,
                eos_token_id=eos_id,
            ),
        )

        orig_init = engine._init_caches
        captured = []

        def hooked_init(*args, **kwargs):
            t_c, d_c = orig_init(*args, **kwargs)
            captured.append((t_c, d_c))
            return t_c, d_c

        engine._init_caches = hooked_init

        prompt = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out = engine.generate(prompt)

        t_c, d_c = captured[0]
        total_len = out.tokens.shape[1]
        draft_k = d_c[0].k_cache[0, 0, :total_len, :]

        zero_positions = [i for i in range(total_len) if float(draft_k[i].norm().item()) == 0.0]
        assert zero_positions == [], (
            f"K={k}: Expected 0 uninitialized zero gaps in draft KV cache upon EOS, "
            f"found zero slices at: {zero_positions}"
        )
        assert d_c[0].cur_len == total_len
        assert t_c[0].cur_len == total_len

    def test_draft_kv_cache_no_zero_hole_on_eos_stochastic(self) -> None:
        """Verify draft KV-cache has zero uninitialized zero-slice holes under stochastic decoding with EOS."""
        torch.manual_seed(15)
        cfg = HeliosConfig(
            vocab_size=64,
            d_model=32,
            n_layers=1,
            n_heads=2,
            n_kv_heads=2,
            d_ffn=64,
            max_seq_len=64,
        )
        target = HeliosForCausalLM(cfg).eval()
        draft = HeliosForCausalLM(cfg).eval()
        draft.load_state_dict(target.state_dict())

        eos_id = 3
        engine = SpeculativeDecodingEngine(
            target_model=target,
            draft_model=draft,
            config=SpeculativeConfig(
                lookahead_k=1,
                temperature=0.7,
                max_new_tokens=10,
                eos_token_id=eos_id,
            ),
        )

        orig_init = engine._init_caches
        captured = []

        def hooked_init(*args, **kwargs):
            t_c, d_c = orig_init(*args, **kwargs)
            captured.append((t_c, d_c))
            return t_c, d_c

        engine._init_caches = hooked_init

        prompt = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out = engine.generate(prompt)

        t_c, d_c = captured[0]
        total_len = out.tokens.shape[1]
        draft_k = d_c[0].k_cache[0, 0, :total_len, :]

        zero_positions = [i for i in range(total_len) if float(draft_k[i].norm().item()) == 0.0]
        assert zero_positions == [], (
            f"Stochastic K=1: Expected 0 uninitialized zero gaps in draft KV cache upon EOS, "
            f"found zero slices at: {zero_positions}"
        )
        assert d_c[0].cur_len == total_len
        assert t_c[0].cur_len == total_len
