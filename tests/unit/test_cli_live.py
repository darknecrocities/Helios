"""Unit and Integration Tests for Helios Live Interactive CLI Commands.

Tests `helios generate`, `helios chat`, and `helios eval`.

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from helios.cli.chat import ChatSession
from helios.cli.main import app
from helios.cli.utils import load_model_and_config, load_or_create_tokenizer

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
