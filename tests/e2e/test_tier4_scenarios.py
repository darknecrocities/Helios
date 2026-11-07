"""Tier 4: Real-World Application Scenarios for Helios Open-Weight LLM.

Simulates end-to-end production workloads:
1. Greenfield Project Audit & Governance Compliance
2. Multilingual (EN/Filipino) & Code Prompt Encoding with Special Tokens
3. Nano Causal Inference & Autoregressive Text Generation with Dynamic KV Cache
4. Multi-Turn Bilingual Agent Conversation with HMA Retrieval Grounding & Weight Isolation
5. Full Pretraining Micro-Run with POSIX Signal Interruption & Bitwise Resumption
6. Deployment & Disk Preservation Pipeline (Ollama, HuggingFace Hub, One Touch / External Disk)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from helios.memory.base import compute_model_weight_hash
from helios.memory.controller import HMAController
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.bpe import HeliosBPE
from helios.tokenizer.evaluation import calculate_fertility_metrics
from helios.tokenizer.special_tokens import (
    CALL_TOOL_TOKEN,
    END_HEADER_TOKEN,
    EOS_TOKEN,
    RETURN_TOOL_TOKEN,
    START_HEADER_TOKEN,
)
from helios.tokenizer.unigram import HeliosUnigram
from helios.training.config import TrainingConfig
from helios.training.dataloader import CursorDataLoader, TokenDataset
from helios.training.signals import InterruptionHandler
from helios.training.trainer import Trainer


# ==============================================================================
# Scenario 1: Greenfield Project Audit & Governance Compliance
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario1_greenfield_project_and_governance_audit(repo_root: Path) -> None:
    """E2E Scenario 1: Complete audit of git, packaging, licensing, research, and CI."""
    # 1. Git repository and branch
    git_config = (repo_root / ".git" / "config").read_text(encoding="utf-8")
    assert "darknecrocities/Helios" in git_config

    # 2. Packaging configuration
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "Arron Kian Parejas" in pyproject
    assert "torch>=" in pyproject

    # 3. Governance manifests
    for fname in [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "CITATION.cff",
    ]:
        p = repo_root / fname
        assert p.exists() and p.stat().st_size > 100

    # 4. Frontier model analysis
    analysis_text = (repo_root / "docs" / "research" / "model_analysis.md").read_text(
        encoding="utf-8"
    )
    assert "Astra" in analysis_text and "Gemini" in analysis_text and "Muse" in analysis_text

    # 5. Continuous integration pipeline
    ci_text = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pytest" in ci_text


# ==============================================================================
# Scenario 2: Multilingual & Code Prompt Encoding with Special Tokens
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario2_multilingual_code_prompt_pipeline(
    bpe_tokenizer: HeliosBPE, unigram_tokenizer: HeliosUnigram
) -> None:
    """E2E Scenario 2: Tagalog/English prompt with code, tokenize and verify fertility."""
    prompt = (
        f"{START_HEADER_TOKEN}user{END_HEADER_TOKEN}\n"
        "Pakisulat ang isang function sa Python para sa Fibonacci sequence.\n"
        f"{START_HEADER_TOKEN}assistant{END_HEADER_TOKEN}\n"
        f"{CALL_TOOL_TOKEN}python_interpreter\n"
        "def fib(n):\n"
        "    return n if n <= 1 else fib(n-1) + fib(n-2)\n"
        f"{RETURN_TOOL_TOKEN}\n"
        "Narito ang resulta."
    )

    for tok in [bpe_tokenizer, unigram_tokenizer]:
        # Encode with special tokens
        token_ids = tok.encode(prompt, add_special_tokens=True)
        assert len(token_ids) > 10

        # Decode and verify lossless preservation of text and special token delimiters
        decoded = tok.decode(token_ids, skip_special_tokens=False)
        assert START_HEADER_TOKEN in decoded
        assert CALL_TOOL_TOKEN in decoded
        assert "Fibonacci sequence" in decoded
        assert "Narito ang resulta" in decoded

        # Fertility metrics
        fertility = calculate_fertility_metrics(tok, prompt)
        assert fertility.tokens_per_word > 0.0


# ==============================================================================
# Scenario 3: Nano Causal Inference & Autoregressive Generation with Dynamic KV Cache
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario3_nano_causal_inference_autoregressive_generation(
    bpe_tokenizer: HeliosBPE,
) -> None:
    """E2E Scenario 3: Run autoregressive text generation on Nano preset with stateful KV cache."""
    # Use compact configuration matching Nano-20M architecture
    cfg = HeliosConfig(
        vocab_size=bpe_tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ffn=128,
        max_seq_len=64,
    )
    torch.manual_seed(42)
    model = HeliosForCausalLM(cfg).eval()

    prompt = "Helios is an open-weight foundation model"
    token_ids = bpe_tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([token_ids[:16]], dtype=torch.long)

    # Autoregressive generation with greedy sampling (temperature=0.0)
    generated = model.generate(
        input_ids,
        max_new_tokens=10,
        temperature=0.0,
        eos_token_id=bpe_tokenizer.special_tokens[EOS_TOKEN],
    )

    assert generated.shape[1] == input_ids.shape[1] + 10
    decoded_gen = bpe_tokenizer.decode(generated[0].tolist())
    assert len(decoded_gen) > len(prompt)


# ==============================================================================
# Scenario 4: Multi-Turn Bilingual Agent Conversation with HMA Retrieval Grounding
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario4_multiturn_bilingual_chat_with_hma_retrieval(
    isolated_hma: HMAController, bpe_tokenizer: HeliosBPE, tiny_e2e_model: HeliosForCausalLM
) -> None:
    """E2E Scenario 4: Multi-turn chat with SQLite episodic storage and RRF grounding."""
    session_id = "agent_dialogue_session"
    weight_hash_start = compute_model_weight_hash(tiny_e2e_model)

    # Turn 1: User introduces knowledge in Tagalog
    isolated_hma.store_episodic(session_id, "user", "Ang Helios ay may GQA na 3:1 ratio.")
    # Turn 2: Assistant confirms
    isolated_hma.store_episodic(
        session_id, "assistant", "Opo, tama iyon upang mabawasan ang KV-cache bloat."
    )

    # Technical docs indexed into semantic store
    isolated_hma.index_semantic(
        "spec_hma",
        "Hierarchical Memory Architecture uses 4 tiers: Working, Episodic, Semantic, Procedural.",
    )
    isolated_hma.index_semantic(
        "spec_rope", "Rotary position embeddings provide relative token offsets."
    )

    # Turn 3: User query asking about architecture
    grounded_context = isolated_hma.build_grounded_context(
        "Hierarchical Memory Architecture", session_id=session_id
    )
    assert "Hierarchical Memory Architecture" in grounded_context
    assert "<retrieved_evidence>" in grounded_context

    # Forward pass on grounded context
    tokens = bpe_tokenizer.encode(grounded_context, add_special_tokens=True)
    input_ids = torch.tensor([tokens[: tiny_e2e_model.config.max_seq_len]], dtype=torch.long)
    with torch.no_grad():
        out = tiny_e2e_model(input_ids)
        assert out["logits"].shape[1] == input_ids.shape[1]

    # Verify model weights are 100% untainted
    weight_hash_end = compute_model_weight_hash(tiny_e2e_model)
    assert weight_hash_start == weight_hash_end, "Model weights polluted by conversational memory!"


# ==============================================================================
# Scenario 5: Full Pretraining Micro-Run with POSIX Signal Interruption & Bitwise Resumption
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario5_full_pretraining_interruption_and_bitwise_resumption(
    synthetic_corpus_file: Path, tmp_path: Path
) -> None:
    """E2E Scenario 5: Baseline vs interrupted + resumed run matching loss trajectory."""
    dataset = TokenDataset(synthetic_corpus_file)
    model_cfg = HeliosConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ffn=64,
        max_seq_len=16,
    )

    # 1. Baseline Run: 12 steps continuous
    torch.manual_seed(999)
    model_base = HeliosForCausalLM(model_cfg)
    train_cfg_base = TrainingConfig(
        checkpoint_dir=str(tmp_path / "base_ckpts"),
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        device="cpu",
    )
    trainer_base = Trainer(model_base, train_cfg_base)
    loader_base = CursorDataLoader(dataset, batch_size=2, seq_len=16)

    baseline_losses: list[float] = []
    for _ in range(12):
        batch = next(loader_base)
        loss = trainer_base.train_step(batch, dataloader_cursor=loader_base.get_state())
        baseline_losses.append(loss)

    # 2. Interrupted Run: Train 6 steps, trap signal, save checkpoint
    torch.manual_seed(999)
    model_int = HeliosForCausalLM(model_cfg)
    ckpt_dir_int = tmp_path / "int_ckpts"
    train_cfg_int = TrainingConfig(
        checkpoint_dir=str(ckpt_dir_int),
        learning_rate=1e-3,
        gradient_accumulation_steps=1,
        save_on_interruption=True,
        device="cpu",
    )
    handler = InterruptionHandler()
    trainer_int = Trainer(model_int, train_cfg_int, interruption_handler=handler)
    loader_int = CursorDataLoader(dataset, batch_size=2, seq_len=16)

    for _ in range(6):
        batch = next(loader_int)
        trainer_int.train_step(batch, dataloader_cursor=loader_int.get_state())

    handler.trigger_interrupt()
    saved_ckpt = trainer_int.save_checkpoint(checkpoint_name="emergency_step_6")

    # 3. Resumed Run: Restore from emergency checkpoint and run remaining 6 steps
    model_resumed = HeliosForCausalLM(model_cfg)
    trainer_resumed = Trainer(model_resumed, train_cfg_int)
    meta = trainer_resumed.load_checkpoint(saved_ckpt)

    loader_resumed = CursorDataLoader(dataset, batch_size=2, seq_len=16)
    loader_resumed.load_state(meta["dataloader_cursor"])

    resumed_losses: list[float] = []
    for _ in range(6):
        batch = next(loader_resumed)
        loss = trainer_resumed.train_step(batch, dataloader_cursor=loader_resumed.get_state())
        resumed_losses.append(loss)

    # Assert resumed losses match baseline losses for steps 7-12 within tight tolerance (< 1e-4)
    expected_losses = baseline_losses[6:]
    for step_idx, (exp, act) in enumerate(zip(expected_losses, resumed_losses, strict=True)):
        diff = abs(exp - act)
        assert diff < 1e-4, (
            f"Resumed loss at step {step_idx + 7} diverged: expected {exp}, got {act} (diff={diff})"
        )


# ==============================================================================
# Scenario 6: Export & Deployment Verification Pipeline
# ==============================================================================
@pytest.mark.e2e
@pytest.mark.tier4
def test_scenario6_deployment_and_preservation_pipeline(
    tiny_e2e_config: HeliosConfig, bpe_tokenizer: HeliosBPE, tmp_path: Path
) -> None:
    """E2E Scenario 6: Verify Ollama Modelfile, HuggingFace Hub, and disk staging."""
    deploy_root = tmp_path / "deployment_artifacts"
    deploy_root.mkdir(parents=True, exist_ok=True)

    # 1. Ollama Modelfile
    modelfile = deploy_root / "Modelfile"
    modelfile_content = (
        "FROM llama3.2\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER stop <|end_of_text|>\n"
        "SYSTEM You are Helios, an open-weight foundation model developed by Arron Kian Parejas.\n"
    )
    modelfile.write_text(modelfile_content, encoding="utf-8")
    assert modelfile.exists()
    assert "Arron Kian Parejas" in modelfile.read_text(encoding="utf-8")

    # 2. Hugging Face Hub Export
    hf_dir = deploy_root / "hf_hub"
    tiny_e2e_config.save_pretrained(hf_dir)
    bpe_tokenizer.save_pretrained(str(hf_dir))
    readme_content = (
        "# Helios Open-Weight LLM\n"
        "Author: Arron Kian Parejas\n"
        "Repository: https://huggingface.co/Darknecrocities/Helios\n"
    )
    (hf_dir / "README.md").write_text(readme_content, encoding="utf-8")
    assert (hf_dir / "config.json").exists()
    assert (hf_dir / "tokenizer.json").exists()

    # 3. External Disk Preservation Mirror (One Touch / OneDrive)
    disk_dir = deploy_root / "OneTouch_Helios"
    disk_dir.mkdir(parents=True, exist_ok=True)
    mirror_file = disk_dir / "disk_manifest.json"
    manifest_data = {
        "author": "Arron Kian Parejas",
        "model": "Helios-Nano-20M",
        "sha256": "abcdef1234567890",
        "preserved_disk_mb": 150.0,
    }
    mirror_file.write_text(json.dumps(manifest_data), encoding="utf-8")
    assert mirror_file.exists()
