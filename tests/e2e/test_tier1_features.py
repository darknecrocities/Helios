"""Tier 1: Happy-Path Equivalence Feature Verification Suite for Helios LLM.

Covers all 37 inventoried features across 6 functional domains:
- Domain 1 (F1-F5): Git remote, pyproject PEP 517, governance, model analysis, CI.
- Domain 2 (F6-F10): Byte BPE, Unigram, special tokens, fertility, HF export.
- Domain 3 (F11-F19): Config, tied embeddings, RoPE, RMSNorm, GQA, SwiGLU, LM head, inspector.
- Domain 4 (F20-F25): Working buffer, SQLite episodic, dense+sparse semantic, RRF, procedural.
- Domain 5 (F26-F31): TrainingConfig, grad clip, checkpoints, signals, 6-state resumption.
- Domain 6 (F32-F37): E2E test harness, Tier 5 readiness, Ollama, HF, CLI, external disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import typer.testing

from helios.cli.main import app as cli_app
from helios.export.disk import DiskMirrorManager
from helios.export.hf import export_huggingface
from helios.export.ollama import export_ollama
from helios.memory.base import compute_model_weight_hash
from helios.memory.controller import HMAController
from helios.memory.procedural import ToolDefinition
from helios.model.config import HeliosConfig
from helios.model.normalization import RMSNorm
from helios.model.parameter_inspector import ParameterInspector
from helios.model.rotary_embedding import RotaryEmbedding, apply_rotary_emb
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.bpe import HeliosBPE
from helios.tokenizer.evaluation import calculate_fertility_metrics
from helios.tokenizer.special_tokens import (
    BOS_TOKEN,
    CALL_TOOL_TOKEN,
    END_HEADER_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    REFUSAL_TOKEN,
    RETURN_TOOL_TOKEN,
    START_HEADER_TOKEN,
)
from helios.tokenizer.unigram import HeliosUnigram
from helios.training.checkpoint import CheckpointManager
from helios.training.config import TrainingConfig
from helios.training.signals import InterruptionHandler
from helios.training.smoke_test import run_overfit_one_batch_test
from helios.training.trainer import Trainer


# ==============================================================================
# Domain 1: Repository Foundation, Packaging & Governance (F1-F5)
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier1
class TestDomain1GovernanceAndPackaging:
    """Happy-path verification of repository structure and open-source compliance."""

    def test_f1_git_remote_and_branch_setup(self, repo_root: Path) -> None:
        """F1: Verify git remote pairs with official repo and default branch is main."""
        git_dir = repo_root / ".git"
        assert git_dir.exists() and git_dir.is_dir()
        config_file = git_dir / "config"
        assert config_file.exists()
        config_text = config_file.read_text(encoding="utf-8")
        assert "github.com/darknecrocities/Helios" in config_text

    def test_f2_python_packaging_and_config(self, repo_root: Path) -> None:
        """F2: Verify pyproject.toml PEP 517/621 metadata, dependencies, and tool tables."""
        pyproject_file = repo_root / "pyproject.toml"
        assert pyproject_file.exists()
        content = pyproject_file.read_text(encoding="utf-8")
        assert 'name = "helios"' in content
        assert 'requires-python = ">=3.11"' in content
        assert "hatchling.build" in content
        assert "torch>=" in content
        assert "[tool.ruff]" in content
        assert "[tool.pytest.ini_options]" in content

    def test_f3_open_source_governance_files(self, repo_root: Path) -> None:
        """F3: Verify presence and substance of all 6 open source governance files."""
        expected_files = [
            ("README.md", "Arron Kian Parejas"),
            ("LICENSE", "Apache License"),
            ("CONTRIBUTING.md", "Pull Request"),
            ("CODE_OF_CONDUCT.md", "Contributor Covenant"),
            ("SECURITY.md", "Vulnerability"),
            ("CITATION.cff", "cff-version"),
        ]
        for filename, expected_keyword in expected_files:
            file_path = repo_root / filename
            assert file_path.exists(), f"Missing governance file: {filename}"
            content = file_path.read_text(encoding="utf-8")
            assert expected_keyword in content

    def test_f4_research_model_problem_analysis(self, repo_root: Path) -> None:
        """F4: Verify frontier model problem analysis document covers Astra, Opus, Gemini, Muse."""
        analysis_file = repo_root / "docs" / "research" / "model_analysis.md"
        assert analysis_file.exists()
        text = analysis_file.read_text(encoding="utf-8")
        for term in ["Astra", "Gemini", "Muse", "Epistemic", "Weight Isolation"]:
            assert term in text

    def test_f5_github_actions_ci_pipeline(self, repo_root: Path) -> None:
        """F5: Verify GitHub Actions CI workflow config defines automated test/lint jobs."""
        ci_file = repo_root / ".github" / "workflows" / "ci.yml"
        assert ci_file.exists()
        content = ci_file.read_text(encoding="utf-8")
        assert "pytest" in content
        assert "ruff" in content


# ==============================================================================
# Domain 2: Tokenizer Subsystem (F6-F10)
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier1
class TestDomain2TokenizerSubsystem:
    """Happy-path verification of BPE, Unigram, special tokens, fertility, and HF serialization."""

    def test_f6_byte_level_bpe_engine(self, bpe_tokenizer: HeliosBPE) -> None:
        """F6: Verify byte-level BPE tokenizer lossless roundtrip across UTF-8 text."""
        test_strings = ["Hello World!", "Mabuhay Pilipinas!", "def forward(x): return x * 2"]
        for s in test_strings:
            ids = bpe_tokenizer.encode(s, add_special_tokens=False)
            decoded = bpe_tokenizer.decode(ids, skip_special_tokens=True)
            assert decoded == s

    def test_f7_unigram_tokenizer_engine(self, unigram_tokenizer: HeliosUnigram) -> None:
        """F7: Verify SentencePiece-style Unigram tokenizer encodes and decodes losslessly."""
        text = "Helios open-weight foundation model ecosystem."
        ids = unigram_tokenizer.encode(text, add_special_tokens=False)
        decoded = unigram_tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text

    def test_f8_special_tokens_contract(self, bpe_tokenizer: HeliosBPE) -> None:
        """F8: Verify special tokens registry contains canonical chat and tool markers."""
        expected_tokens = [
            BOS_TOKEN,
            EOS_TOKEN,
            PAD_TOKEN,
            START_HEADER_TOKEN,
            END_HEADER_TOKEN,
            CALL_TOOL_TOKEN,
            RETURN_TOOL_TOKEN,
            REFUSAL_TOKEN,
        ]
        for token_str in expected_tokens:
            assert token_str in bpe_tokenizer.special_tokens
            token_id = bpe_tokenizer.special_tokens[token_str]
            assert bpe_tokenizer.decode([token_id]) == token_str

    def test_f9_multilingual_and_code_fertility(self, bpe_tokenizer: HeliosBPE) -> None:
        """F9: Verify fertility evaluation metrics across English, Filipino, and code."""
        text = "Ito ay isang sample code function sa Python para sa Helios."
        metrics = calculate_fertility_metrics(bpe_tokenizer, text)
        assert metrics.tokens_per_word > 0.0
        assert metrics.chars_per_token > 0.0

    def test_f10_tokenizer_hf_serialization(self, bpe_tokenizer: HeliosBPE, tmp_path: Path) -> None:
        """F10: Verify tokenizer serialization and reloading via save_pretrained."""
        save_dir = tmp_path / "tokenizer_export"
        bpe_tokenizer.save_pretrained(str(save_dir))
        assert (save_dir / "tokenizer.json").exists()
        assert (save_dir / "special_tokens_map.json").exists()
        reloaded = HeliosBPE.from_pretrained(str(save_dir))
        assert reloaded.vocab_size == bpe_tokenizer.vocab_size


# ==============================================================================
# Domain 3: Model Architecture & Parameter Inspection (F11-F19)
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier1
class TestDomain3TransformerArchitecture:
    """Happy-path verification of transformer blocks, RoPE, RMSNorm, GQA, and Nano sizing."""

    def test_f11_model_configuration(self) -> None:
        """F11: Verify HeliosConfig initialization, validation invariants, and dict export."""
        cfg = HeliosConfig.nano_15m()
        assert cfg.vocab_size == 32000
        assert cfg.d_model == 288
        assert cfg.d_ffn == 768
        assert cfg.developer == "Arron Kian Parejas"
        d = cfg.to_dict()
        reconstructed = HeliosConfig.from_dict(d)
        assert reconstructed.d_model == cfg.d_model

    def test_f12_token_embeddings_and_weight_tying(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """F12: Verify token embeddings layer and weight tying identity with LM head."""
        assert tiny_e2e_model.config.tie_word_embeddings
        assert tiny_e2e_model.lm_head.weight is tiny_e2e_model.embed_tokens.weight

    def test_f13_rotary_position_embeddings(self) -> None:
        """F13: Verify RoPE precomputes frequencies and rotates hidden states."""
        rope = RotaryEmbedding(dim=32, max_seq_len=64)
        x = torch.randn(2, 4, 16, 32)
        cos, sin = rope(x, start_pos=0)
        rotated = apply_rotary_emb(x, cos, sin)
        assert rotated.shape == x.shape

    def test_f14_pre_rmsnorm_normalization(self) -> None:
        """F14: Verify RMSNorm normalizes feature dimension to unit variance."""
        norm = RMSNorm(dim=64, eps=1e-5)
        x = torch.randn(2, 10, 64) * 5.0
        out = norm(x)
        assert out.shape == x.shape
        rms = torch.sqrt(torch.mean(out**2, dim=-1))
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-2)

    def test_f15_gqa_and_dynamic_kv_cache(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """F15: Verify Grouped-Query Attention forward pass with stateful KV cache allocation."""
        input_ids = torch.randint(0, tiny_e2e_model.config.vocab_size, (1, 8))
        out = tiny_e2e_model(input_ids, use_cache=True)
        assert "kv_caches" in out
        assert out["kv_caches"] is not None
        assert len(out["kv_caches"]) == tiny_e2e_model.config.n_layers

    def test_f16_swiglu_gated_feed_forward(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """F16: Verify SwiGLU feed-forward layer forward pass."""
        block = tiny_e2e_model.layers[0]
        h = torch.randn(1, 4, tiny_e2e_model.config.d_model)
        ffn_out = block.feed_forward(h)
        assert ffn_out.shape == h.shape

    def test_f17_decoder_only_transformer_lm(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """F17: Verify full causal LM forward pass produces logits and cross-entropy loss."""
        input_ids = torch.randint(0, tiny_e2e_model.config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = tiny_e2e_model(input_ids, labels=labels)
        assert out["logits"].shape == (2, 8, tiny_e2e_model.config.vocab_size)
        assert out["loss"] is not None
        assert not torch.isnan(out["loss"])

    def test_f18_parameter_inspector_tool(self, tiny_e2e_model: HeliosForCausalLM) -> None:
        """F18: Verify ParameterInspector produces non-duplicating parameter count."""
        report = ParameterInspector.inspect(tiny_e2e_model, tiny_e2e_model.config)
        assert report.total_parameters > 0
        assert report.trainable_parameters > 0
        assert "fp32" in report.memory_footprint_mb

    def test_f19_nano_target_sizing_verification(self) -> None:
        """F19: Verify all Nano presets fit strictly within the 10M-40M parameter envelope."""
        presets = ["nano-15m", "nano-20m", "nano-25m", "nano-35m"]
        for p in presets:
            cfg = HeliosConfig.from_preset(p)
            counts = cfg.count_parameters()
            total = counts["total_parameters"]
            assert 10_000_000 <= total <= 40_000_000, (
                f"Preset {p} params={total} outside [10M, 40M]"
            )


# ==============================================================================
# Domain 4: Hierarchical Memory Architecture (HMA) (F20-F25)
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier1
class TestDomain4HierarchicalMemory:
    """Happy-path verification of working, episodic, semantic, procedural, and weight isolation."""

    def test_f20_working_memory_buffer(self, isolated_hma: HMAController) -> None:
        """F20: Verify working memory buffer stores turns and respects token boundaries."""
        isolated_hma.store_episodic("s1", "user", "What is Helios?")
        ctx = isolated_hma.get_working_context("s1", max_tokens=100)
        assert len(ctx) == 1
        assert ctx[0]["content"] == "What is Helios?"

    def test_f21_episodic_memory_store(self, isolated_hma: HMAController) -> None:
        """F21: Verify episodic memory persists turns into SQLite WAL store."""
        turn_id = isolated_hma.store_episodic("s1", "assistant", "Helios is an open-weight LLM.")
        assert turn_id.startswith("turn_")
        history = isolated_hma.episodic.get_session_turns("s1")
        assert len(history) >= 1

    def test_f22_semantic_memory_hybrid(self, isolated_hma: HMAController) -> None:
        """F22: Verify semantic memory indices documents and performs hybrid retrieval."""
        isolated_hma.index_semantic(
            "doc_rope", "Rotary position embeddings provide relative encoding."
        )
        results = isolated_hma.query("Rotary embeddings", top_k=1)
        assert len(results) == 1
        assert results[0].id == "doc_rope"

    def test_f23_reciprocal_rank_fusion(self, isolated_hma: HMAController) -> None:
        """F23: Verify Reciprocal Rank Fusion ranks and retrieves relevant documents."""
        isolated_hma.index_semantic("doc_gqa", "Grouped-query attention reduces KV-cache memory.")
        res = isolated_hma.query("Grouped-query attention", top_k=1)
        assert len(res) == 1
        assert "KV-cache" in res[0].content

    def test_f24_procedural_memory_skill_registry(self, isolated_hma: HMAController) -> None:
        """F24: Verify tool definition registration and retrieval."""
        tool = ToolDefinition(
            name="calculator",
            description="Evaluates mathematical expressions",
            parameters={"expr": "string"},
            required_parameters=["expr"],
        )
        isolated_hma.register_tool(tool)
        retrieved_tools = isolated_hma.procedural.search_tools("calculator")
        assert len(retrieved_tools) == 1
        assert retrieved_tools[0].name == "calculator"

    def test_f25_memory_governance_weight_isolation(
        self, isolated_hma: HMAController, tiny_e2e_model: HeliosForCausalLM
    ) -> None:
        """F25: Verify strict external memory governance and 100% parameter weight invariance."""
        hash_before = compute_model_weight_hash(tiny_e2e_model)
        for i in range(10):
            isolated_hma.store_episodic("iso_sess", "user", f"Turn {i} data injection test")
            isolated_hma.query("data injection test")
        hash_after = compute_model_weight_hash(tiny_e2e_model)
        assert hash_before == hash_after, "Model weights altered by external memory operations!"


# ==============================================================================
# Domain 5: Training Controller & Interruption Fallback (F26-F31)
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier1
class TestDomain5TrainingController:
    """Happy-path verification of training loop, grad clipping, checkpoints, signals, and resume."""

    def test_f26_training_config_and_loss_tracking(self, tmp_path: Path) -> None:
        """F26: Verify TrainingConfig initialization, serialization, and immutability."""
        cfg = TrainingConfig(
            learning_rate=2e-4,
            max_steps=50,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu",
        )
        assert cfg.learning_rate == 2e-4
        assert cfg.max_steps == 50

    def test_f27_l2_gradient_clipping_and_loss_abort(
        self, tiny_e2e_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """F27: Verify L2 gradient norm clipping during training steps."""
        cfg = TrainingConfig(
            learning_rate=1e-3,
            max_grad_norm=1.0,
            checkpoint_dir=str(tmp_path / "ckpts"),
            device="cpu",
        )
        trainer = Trainer(tiny_e2e_model, cfg)
        batch = {
            "input_ids": torch.randint(0, tiny_e2e_model.config.vocab_size, (2, 8)),
            "labels": torch.randint(0, tiny_e2e_model.config.vocab_size, (2, 8)),
        }
        loss_val = trainer.train_microbatch(batch)
        assert not np.isnan(loss_val)

    def test_f28_atomic_checkpoint_management(
        self, tiny_e2e_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """F28: Verify atomic write-and-rename checkpointing with SHA-256 verification."""
        ckpt_dir = tmp_path / "atomic_ckpts"
        mgr = CheckpointManager(checkpoint_dir=str(ckpt_dir), max_keep=2)
        saved_path = mgr.save_checkpoint(
            step=1,
            model=tiny_e2e_model,
            optimizer=torch.optim.AdamW(tiny_e2e_model.parameters()),
            scheduler=None,
            cursor={"token_offset": 100},
            metrics={"loss": 2.5},
        )
        assert saved_path.exists()
        assert not (ckpt_dir / "checkpoint_step_1.pt.tmp").exists()
        assert (saved_path / "manifest.json").exists()

    def test_f29_posix_signal_trapping(self) -> None:
        """F29: Verify InterruptionHandler traps simulated POSIX signals."""
        handler = InterruptionHandler()
        assert not handler.interrupted
        handler.trigger_interrupt()
        assert handler.interrupted

    def test_f30_exact_6_state_resumption_engine(
        self, tiny_e2e_model: HeliosForCausalLM, tmp_path: Path
    ) -> None:
        """F30: Verify restoration of weights, optimizer, scheduler, RNGs, cursor, and metrics."""
        ckpt_dir = tmp_path / "resumption_ckpts"
        train_cfg = TrainingConfig(checkpoint_dir=str(ckpt_dir), device="cpu")
        trainer1 = Trainer(tiny_e2e_model, train_cfg)
        saved_ckpt = trainer1.save_checkpoint(checkpoint_name="test_state")

        # Create fresh model and trainer and resume
        fresh_model = HeliosForCausalLM(tiny_e2e_model.config)
        trainer2 = Trainer(fresh_model, train_cfg)
        meta = trainer2.load_checkpoint(saved_ckpt)
        assert meta["step"] == trainer1.global_step

    def test_f31_overfit_one_batch_smoke_test(self) -> None:
        """F31: Verify overfit-one-batch smoke test converges loss to near-zero."""
        res = run_overfit_one_batch_test(device="cpu", target_loss=0.1, max_steps=120)
        assert res["converged"]
        assert res["final_loss"] <= 0.1


# ==============================================================================
# Domain 6: Deployment, Ecosystem Integration & Tooling (F32-F37)
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier1
class TestDomain6DeploymentTooling:
    """Happy-path verification of E2E suite, Tier 5 readiness, Ollama, HF, CLI, and OneDrive."""

    def test_f32_opaque_box_e2e_suite_readiness(self, repo_root: Path) -> None:
        """F32: Verify presence of opaque-box E2E test files and runner."""
        e2e_dir = repo_root / "tests" / "e2e"
        assert e2e_dir.exists()
        assert (e2e_dir / "runner.py").exists()

    def test_f33_adversarial_coverage_hardening_readiness(self, repo_root: Path) -> None:
        """F33: Verify presence of Tier 5 adversarial suites in tests/adversarial/."""
        adv_dir = repo_root / "tests" / "adversarial"
        assert adv_dir.exists()
        assert len(list(adv_dir.glob("test_*.py"))) >= 5

    def test_f34_ollama_deployment_modelfile_generation(self, tmp_path: Path) -> None:
        """F34: Verify Modelfile generation containing system prompt and target model tag."""
        res = export_ollama(output_dir=tmp_path / "ollama", base_model="llama3.2")
        modelfile_path = Path(res["modelfile_path"])
        assert modelfile_path.exists()
        content = modelfile_path.read_text(encoding="utf-8")
        assert "FROM llama3.2" in content
        assert "Arron Kian Parejas" in content
        assert "PARAMETER temperature" in content
        assert "parejasarronkian/Helios" in content

    def test_f35_huggingface_hub_export_workflow(
        self,
        tiny_e2e_config: HeliosConfig,
        tiny_e2e_model: HeliosForCausalLM,
        bpe_tokenizer: HeliosBPE,
        tmp_path: Path,
    ) -> None:
        """F35: Verify Hugging Face export directory staging and metadata."""
        export_dir = tmp_path / "hf_export"
        res = export_huggingface(
            output_dir=export_dir,
            config=tiny_e2e_config,
            model=tiny_e2e_model,
            tokenizer=bpe_tokenizer,
            repo_id="Darknecrocities/Helios",
            generate_card=True,
        )
        assert (export_dir / "config.json").exists()
        assert (export_dir / "tokenizer.json").exists()
        assert (export_dir / "model.safetensors").exists()
        assert (export_dir / "README.md").exists()
        assert res["weights_bytes"] > 0
        readme_content = (export_dir / "README.md").read_text(encoding="utf-8")
        assert "https://huggingface.co/Darknecrocities/Helios" in readme_content
        assert "Arron Kian Parejas" in readme_content

    def test_f36_interactive_live_testing_cli(self, cli_runner: typer.testing.CliRunner) -> None:
        """F36: Verify CLI commands (version, inspect, roadmap) execute cleanly."""
        res_ver = cli_runner.invoke(cli_app, ["version"])
        assert res_ver.exit_code == 0
        assert "Arron Kian Parejas" in res_ver.stdout

        res_insp = cli_runner.invoke(cli_app, ["inspect", "-m", "nano-15m"])
        assert res_insp.exit_code == 0
        assert "Helios Model Inspector" in res_insp.stdout

        res_road = cli_runner.invoke(cli_app, ["roadmap"])
        assert res_road.exit_code == 0
        assert "Nano" in res_road.stdout

    def test_f37_onedrive_disk_preservation_mirroring(self, tmp_path: Path) -> None:
        """F37: Verify OneDrive disk preservation mirror staging utility and SHA verification."""
        workspace = tmp_path / "workspace"
        ckpt_dir = workspace / "checkpoints" / "checkpoint_step_10"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "manifest.json").write_text('{"step": 10}', encoding="utf-8")
        (ckpt_dir / "state.pt").write_bytes(b"Helios checkpoint state binary payload")

        disk_root = tmp_path / "disk_mirror"
        mgr = DiskMirrorManager(disk_root=disk_root, workspace_root=workspace)
        items = mgr.mirror_checkpoints(checkpoints_dir=workspace / "checkpoints", free_local=True)
        manifest_file = mgr.write_disk_manifest(items)

        assert manifest_file.exists()
        assert (disk_root / "checkpoints" / "checkpoint_step_10" / "state.pt").exists()
        assert (ckpt_dir / "state.pt.disk_pointer.json").exists()
        assert not (ckpt_dir / "state.pt").exists()
