"""Unit Tests for In-Chat Slash Commands Router and Handlers (Feature 39).

Validates command parsing, whitespace normalization, CoT toggle state transitions,
memory/stats inspection, unknown command interception, and clean exit signals.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from helios.cli.chat import ChatSession
from helios.cli.commands import (
    CommandResult,
    dispatch_chat_command,
    parse_slash_command,
)
from helios.cli.utils import load_model_and_config, load_or_create_tokenizer


@pytest.mark.unit
def test_parse_slash_command_tokenization() -> None:
    """Verify robust tokenization, whitespace normalization, and casing."""
    # Standard slash commands
    assert parse_slash_command("/models") == ("/models", "")
    assert parse_slash_command("  /MODELS  ") == ("/models", "")
    assert parse_slash_command("/help") == ("/help", "")
    assert parse_slash_command("/exit") == ("/exit", "")
    assert parse_slash_command("/quit") == ("/quit", "")
    assert parse_slash_command("/reset") == ("/reset", "")
    assert parse_slash_command("/clear") == ("/clear", "")
    assert parse_slash_command("/memory") == ("/memory", "")
    assert parse_slash_command("/stats") == ("/stats", "")

    # Arguments with varying whitespace and casing
    assert parse_slash_command("/model alpha") == ("/model", "alpha")
    assert parse_slash_command("  /model   ALPHA  ") == ("/model", "ALPHA")
    assert parse_slash_command("/switch \t flagship") == ("/switch", "flagship")
    assert parse_slash_command("  /think    on  ") == ("/think", "on")
    assert parse_slash_command("/think off") == ("/think", "off")
    assert parse_slash_command("/think") == ("/think", "")
    assert parse_slash_command("/unknown_command  arg1  arg2") == ("/unknown_command", "arg1  arg2")

    # Non-slash conversational text must return None
    assert parse_slash_command("What is /model?") is None
    assert parse_slash_command("Hello Helios") is None
    assert parse_slash_command("") is None
    assert parse_slash_command("   ") is None


@pytest.mark.unit
def test_dispatch_chat_command_execution(tmp_path: Path) -> None:
    """Verify all slash commands execute instantaneously and unknown commands are intercepted."""
    model, config = load_model_and_config(preset="nano-15m", device="cpu")
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id="cmd_test",
        db_path=tmp_path / "cmd_test.db",
        device=torch.device("cpu"),
        preset="nano-15m",
    )

    # 1. /think without args displays current state without changing it
    orig_thought = session.show_thought
    res = dispatch_chat_command("/think", session)
    assert res == CommandResult(handled=True, should_exit=False)
    assert session.show_thought == orig_thought

    # 2. /think on / off / toggle cleanly transitions state
    dispatch_chat_command("  /think   off  ", session)
    assert session.show_thought is False
    dispatch_chat_command("/think on", session)
    assert session.show_thought is True
    dispatch_chat_command("/think toggle", session)
    assert session.show_thought is False
    dispatch_chat_command("/think toggle", session)
    assert session.show_thought is True

    # 3. /think with invalid argument outputs syntax guidance without raising
    res_bad_think = dispatch_chat_command("/think invalid_arg", session)
    assert res_bad_think == CommandResult(handled=True, should_exit=False)

    # 4. /models renders catalog
    res_models = dispatch_chat_command("/models", session)
    assert res_models == CommandResult(handled=True, should_exit=False)

    # 5. /model and /switch without arguments display catalog
    res_model_empty = dispatch_chat_command("/model", session)
    assert res_model_empty == CommandResult(handled=True, should_exit=False)
    res_switch_empty = dispatch_chat_command("/switch", session)
    assert res_switch_empty == CommandResult(handled=True, should_exit=False)

    # 6. /model with invalid tier name catches error and does not crash
    res_bad_tier = dispatch_chat_command("/model invalid_tier_xyz", session)
    assert res_bad_tier == CommandResult(handled=True, should_exit=False)
    assert session.preset == "nano-15m"

    # 7. /memory and /stats execute without calling LLM
    turns_before = session.total_turns
    res_mem = dispatch_chat_command("/memory", session)
    assert res_mem == CommandResult(handled=True, should_exit=False)
    res_stat = dispatch_chat_command("/stats", session)
    assert res_stat == CommandResult(handled=True, should_exit=False)
    assert session.total_turns == turns_before

    # 8. Unknown commands are intercepted without calling LLM
    res_unk = dispatch_chat_command("/some_random_typo_command", session)
    assert res_unk == CommandResult(handled=True, should_exit=False)
    assert session.total_turns == turns_before

    # 9. /help renders command reference
    res_help = dispatch_chat_command("/help", session)
    assert res_help == CommandResult(handled=True, should_exit=False)

    # 10. /reset clears session
    session.total_turns = 5
    res_reset = dispatch_chat_command("/reset", session)
    assert res_reset == CommandResult(handled=True, should_exit=False)
    assert session.total_turns == 0

    # 11. /exit and /quit trigger should_exit=True
    assert dispatch_chat_command("/exit", session) == CommandResult(handled=True, should_exit=True)
    assert dispatch_chat_command("/quit", session) == CommandResult(handled=True, should_exit=True)

    # 12. Non-slash conversational input returns handled=False
    assert dispatch_chat_command("What is the speed of light?", session) == CommandResult(
        handled=False, should_exit=False
    )
