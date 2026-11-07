"""Shared Pytest Fixtures for Helios Opaque-Box E2E Test Suite (Tiers 1-4).

Provides lightweight, deterministic, CPU-safe test fixtures for model configuration,
tokenizers, HMA memory controller, streaming dataloaders, and temporary test directories.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import typer.testing

from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory
from helios.memory.semantic import SemanticMemory
from helios.memory.working import WorkingMemoryManager
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
from helios.tokenizer.bpe import HeliosBPE
from helios.tokenizer.unigram import HeliosUnigram


@pytest.fixture
def cli_runner() -> typer.testing.CliRunner:
    """Typer CLI runner for exercising CLI entry points."""
    return typer.testing.CliRunner()


@pytest.fixture
def tiny_e2e_config() -> HeliosConfig:
    """Minimal HeliosConfig for fast, deterministic E2E test execution on CPU."""
    return HeliosConfig(
        vocab_size=264,  # Matches HeliosBPE base byte vocab (256 + 8 special)
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # GQA 2:1 ratio
        d_ffn=128,
        max_seq_len=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
    )


@pytest.fixture
def tiny_e2e_model(tiny_e2e_config: HeliosConfig) -> HeliosForCausalLM:
    """Deterministic HeliosForCausalLM model instance on CPU."""
    torch.manual_seed(42)
    model = HeliosForCausalLM(tiny_e2e_config)
    model.eval()
    return model


@pytest.fixture
def bpe_tokenizer() -> HeliosBPE:
    """Standard Byte-Level BPE tokenizer with default special tokens."""
    return HeliosBPE()


@pytest.fixture
def unigram_tokenizer() -> HeliosUnigram:
    """Standard Unigram tokenizer with default special tokens."""
    return HeliosUnigram()


@pytest.fixture
def isolated_hma(tmp_path: Path) -> HMAController:
    """Isolated HMAController with dedicated SQLite database and empty working memory."""
    db_path = tmp_path / "e2e_episodic.db"
    episodic = EpisodicMemory(db_path=str(db_path))
    working = WorkingMemoryManager(default_max_tokens=512)
    semantic = SemanticMemory()
    return HMAController(
        working_memory=working,
        episodic_memory=episodic,
        semantic_memory=semantic,
        default_session_id="e2e_session",
    )


@pytest.fixture
def synthetic_corpus_file(tmp_path: Path) -> Path:
    """Pre-generated uint16 binary token stream for dataloader testing."""
    token_path = tmp_path / "synthetic_corpus.bin"
    np.random.seed(42)
    # Generate 5,000 synthetic tokens within vocab range [10, 250]
    tokens = np.random.randint(10, 250, size=5000, dtype=np.uint16)
    tokens.tofile(token_path)
    return token_path


@pytest.fixture
def repo_root() -> Path:
    """Absolute Path to the root of the Helios repository."""
    return Path(__file__).resolve().parent.parent.parent
