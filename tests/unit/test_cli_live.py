"""Unit and Integration Tests for Helios Live Interactive CLI Commands.

Tests `helios generate`, `helios chat`, and `helios eval`.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from typer.testing import CliRunner

from helios.cli.chat import ChatSession
from helios.cli.main import app
from helios.cli.utils import (
    check_memory_capacity,
    free_hardware_memory,
    get_system_available_memory_mb,
    get_system_total_memory_mb,
    load_model_and_config,
    load_or_create_tokenizer,
)

runner = CliRunner()


@pytest.mark.unit
def test_cli_generate_code_and_qa() -> None:
    """Verify that helios generate executes code generation and QA prompts."""
    # 1. Code generation prompt
    res_code = runner.invoke(
        app,
        [
            "generate",
            "--prompt",
            "def binary_search(arr, target):",
            "--max-new-tokens",
            "8",
            "--temperature",
            "0.0",
            "--preset",
            "nano-15m",
            "--device",
            "cpu",
            "--no-stream",
        ],
    )
    assert res_code.exit_code == 0
    assert "Response:" in res_code.stdout
    assert "Helios Generation Telemetry" in res_code.stdout
    assert "Tokens Generated" in res_code.stdout
    assert "Throughput" in res_code.stdout

    # 2. Technical QA prompt
    res_qa = runner.invoke(
        app,
        [
            "generate",
            "--prompt",
            "What is the memory hierarchy of Helios?",
            "--max-new-tokens",
            "8",
            "--temperature",
            "0.7",
            "--top-k",
            "20",
            "--preset",
            "nano-15m",
            "--device",
            "cpu",
            "--stream",
        ],
    )
    assert res_qa.exit_code == 0
    assert "Prompt: What is the memory hierarchy of Helios?" in res_qa.stdout
    assert "KV Cache Footprint" in res_qa.stdout


@pytest.mark.unit
def test_cli_eval_benchmarks() -> None:
    """Verify that helios eval executes speed, memory footprint, and perplexity tests."""
    res = runner.invoke(
        app,
        [
            "eval",
            "--preset",
            "nano-15m",
            "--device",
            "cpu",
            "--prompt-len",
            "16",
            "--gen-len",
            "16",
        ],
    )
    assert res.exit_code == 0
    assert "Helios Performance & Perplexity Report" in res.stdout
    assert "Prefill Latency" in res.stdout
    assert "Decode Latency" in res.stdout
    assert "KV Cache Layers" in res.stdout
    assert "Total KV Footprint" in res.stdout
    assert "Perplexity (PPL)" in res.stdout


@pytest.mark.unit
def test_chat_session_commands_and_grounding(tmp_path: Path) -> None:
    """Verify ChatSession state machine, slash commands, and memory grounding."""
    model, config = load_model_and_config(preset="nano-15m", device="cpu")
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
    db_file = tmp_path / "test_episodic.db"

    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id="test_sess",
        db_path=db_file,
        device=next(model.parameters()).device,
    )

    # 1. Turn generation
    reply = session.handle_turn(
        user_input="Hello Helios!",
        max_new_tokens=6,
        stream=False,
    )
    assert isinstance(reply, str)
    assert session.total_turns == 1
    assert session.total_generated_tokens > 0

    # 2. Check memory status
    mem_status = session.memory_status()
    assert "Hierarchical Memory Architecture Status" in mem_status
    assert "Episodic Turns (SQLite)" in mem_status
    assert "Working Memory Turns" in mem_status

    # 3. Check stats status
    stats = session.stats_status()
    assert "Helios Chat Performance Telemetry" in stats
    assert "Total User Turns" in stats
    assert "Average Throughput" in stats

    # 4. Check clear
    session.clear()
    assert session.total_turns == 0


@pytest.mark.unit
def test_cli_chat_interactive_loop(tmp_path: Path) -> None:
    """Verify that helios chat CLI handles slash commands and exits cleanly."""
    db_file = tmp_path / "test_chat_cli.db"
    # Simulate user sending: /help, /memory, /stats, /reset, /exit
    inputs = "/help\n/memory\n/stats\n/reset\n/exit\n"

    res = runner.invoke(
        app,
        [
            "chat",
            "--preset",
            "nano-15m",
            "--device",
            "cpu",
            "--db-path",
            str(db_file),
        ],
        input=inputs,
    )
    assert res.exit_code == 0
    assert "Helios Foundation Interactive Terminal Chat" in res.stdout
    assert "Special Chat Commands:" in res.stdout
    assert "Hierarchical Memory Architecture Status" in res.stdout
    assert "Helios Chat Performance Telemetry" in res.stdout
    assert "Session working memory context cleared" in res.stdout
    assert "Exiting Helios Chat. Goodbye!" in res.stdout


@pytest.mark.unit
def test_system_memory_inspector() -> None:
    """Verify hardware memory inspection utilities return positive values without error."""
    avail = get_system_available_memory_mb()
    total = get_system_total_memory_mb()
    assert isinstance(avail, float)
    assert isinstance(total, float)
    assert avail > 0.0
    assert total >= avail


@pytest.mark.unit
def test_check_memory_capacity() -> None:
    """Verify memory capacity check validates bounds correctly across tiers."""
    ok, msg = check_memory_capacity("nano")
    assert ok is True
    assert "Memory check passed" in msg

    # Mock low memory scenario
    with patch("helios.cli.utils.get_system_available_memory_mb", return_value=100.0):
        ok_low, msg_low = check_memory_capacity("core")
        assert ok_low is False
        assert "Insufficient memory to switch to Core" in msg_low
        assert "Active model remains unchanged" in msg_low


@pytest.mark.unit
def test_free_hardware_memory() -> None:
    """Verify synchronous hardware cache reclamation executes cleanly on CPU/MPS/CUDA."""
    free_hardware_memory()


@pytest.mark.unit
def test_switch_model_transactional_rollback(tmp_path: Path) -> None:
    """Verify that ChatSession rolls back cleanly without losing active model on reload failure."""
    model, config = load_model_and_config(preset="nano-15m", device="cpu")
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
    db_file = tmp_path / "test_rollback.db"

    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id="rollback_test",
        db_path=db_file,
        device=torch.device("cpu"),
        preset="nano-15m",
    )
    orig_model = session.model
    orig_preset = session.preset

    # Simulate catastrophic reload failure (e.g. simulated OOM / MPS kernel crash)
    with patch("helios.cli.chat.reload_model_for_tier", side_effect=RuntimeError("Simulated OOM")):
        with pytest.raises(RuntimeError, match="Simulated OOM"):
            session.switch_model("alpha-100m")

    # Assert model reference and preset were preserved intact
    assert session.model is orig_model
    assert session.preset == orig_preset

    # Verify model is still functional after rollback
    reply = session.handle_turn("Ping", max_new_tokens=4, stream=False)
    assert isinstance(reply, str)
    assert session.total_turns == 1


@pytest.mark.unit
def test_switch_model_insufficient_memory_safeguard(tmp_path: Path) -> None:
    """Verify that insufficient RAM check prevents reload and retains active model."""
    model, config = load_model_and_config(preset="nano-15m", device="cpu")
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
    db_file = tmp_path / "test_mem_guard.db"

    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id="guard_test",
        db_path=db_file,
        device=torch.device("cpu"),
        preset="nano-15m",
    )
    orig_model = session.model

    with patch("helios.cli.chat.check_memory_capacity", return_value=(False, "RAM Exceeded")):
        with pytest.raises(MemoryError, match="RAM Exceeded"):
            session.switch_model("flagship-6b")

    assert session.model is orig_model
    assert session.preset == "nano-15m"


@pytest.mark.unit
def test_live_multi_tier_switching_all_presets(tmp_path: Path) -> None:
    """Verify live switching transitions across presets (nano -> alpha -> beta)."""
    model, config = load_model_and_config(preset="nano-20m", device="cpu")
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
    db_file = tmp_path / "test_multi_tier.db"

    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id="multi_tier_test",
        db_path=db_file,
        device=torch.device("cpu"),
        preset="nano-20m",
    )

    # Switch to Alpha
    res_alpha = session.switch_model("alpha")
    assert "Helios Alpha" in res_alpha
    assert session.preset == "alpha-100m"

    # Switch to Beta
    res_beta = session.switch_model("beta")
    assert "Helios Beta" in res_beta
    assert session.preset == "beta-300m"

    # Same-tier switch is instantaneous no-op
    res_noop = session.switch_model("beta")
    assert "already Helios Beta" in res_noop
    assert session.preset == "beta-300m"
