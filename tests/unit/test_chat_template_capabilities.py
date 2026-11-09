"""Unit tests for Helios Multi-Capability Chat Template and Special Tokens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from helios.tokenizer.base import BaseTokenizer
from helios.tokenizer.bpe import ByteLevelBPETokenizer
from helios.tokenizer.special_tokens import (
    CAPABILITY_SPECIAL_TOKENS,
    IMAGE_TOKEN,
    MEMORY_END_TOKEN,
    MEMORY_START_TOKEN,
    THOUGHT_END_TOKEN,
    THOUGHT_START_TOKEN,
)


@pytest.mark.unit
def test_capability_tokens_registration_and_atomic_encoding(tmp_path: Path) -> None:
    """Verify capability special tokens register atomically and round-trip via save_pretrained."""
    tok = ByteLevelBPETokenizer(model_max_length=262144)
    tok.register_capability_tokens()

    for cap_tok in CAPABILITY_SPECIAL_TOKENS:
        assert cap_tok in tok.special_tokens
        ids = tok.encode(cap_tok, add_special_tokens=False)
        assert len(ids) == 1
        assert tok.decode(ids, skip_special_tokens=False) == cap_tok
        assert tok.decode(ids, skip_special_tokens=True) == ""

    save_dir = tmp_path / "cap_tok"
    tok.save_pretrained(save_dir)

    cfg = json.loads((save_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert cfg["model_max_length"] == 262144
    assert MEMORY_START_TOKEN in cfg["chat_template"]
    assert IMAGE_TOKEN in cfg["chat_template"]
    assert THOUGHT_START_TOKEN in cfg["chat_template"]
    assert "<|call:tool|>" in cfg["chat_template"]
    assert "<|refusal|>" in cfg["chat_template"]

    reloaded = BaseTokenizer.from_pretrained(save_dir)
    assert reloaded.special_tokens == tok.special_tokens


@pytest.mark.unit
def test_apply_chat_template_all_capabilities() -> None:
    """Verify apply_chat_template renders HMA memory, MCP tools, Vision, CoT, and Refusal."""
    tok = ByteLevelBPETokenizer()
    tok.register_capability_tokens()

    tools = [
        {
            "name": "calculator",
            "description": "Evaluate arithmetic expressions deterministically.",
        }
    ]
    messages = [
        {
            "role": "user",
            "memory_context": "Semantic Fact: Luzon is the largest island in the Philippines.",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Analyze this map of Luzon and compute 12 * 12."},
            ],
        },
        {
            "role": "assistant",
            "thought": "First identify Luzon from HMA memory, then invoke calculator(12*12).",
            "content": "Let me calculate that for you.",
            "tool_calls": [{"name": "calculator", "arguments": {"expr": "12*12"}}],
        },
        {
            "role": "tool",
            "content": {"result": 144},
        },
    ]

    rendered = tok.apply_chat_template(
        messages=messages,
        tools=tools,
        hma_context="Working Memory: Active session initialized.",
        epistemic_guard=True,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=False,
    )
    assert isinstance(rendered, str)
    assert rendered.startswith("<|begin_of_text|>")
    assert f"{MEMORY_START_TOKEN}\nWorking Memory: Active session initialized.\n{MEMORY_END_TOKEN}" in rendered
    assert "# Helios MCP Procedural Tools" in rendered
    assert "- calculator: Evaluate arithmetic expressions deterministically." in rendered
    assert "[Helios Epistemic Guard:" in rendered
    assert f"{MEMORY_START_TOKEN}\nSemantic Fact: Luzon is the largest island in the Philippines.\n{MEMORY_END_TOKEN}" in rendered
    assert f"{IMAGE_TOKEN}Analyze this map of Luzon and compute 12 * 12." in rendered
    assert f"{THOUGHT_START_TOKEN}\nFirst identify Luzon from HMA memory, then invoke calculator(12*12).\n{THOUGHT_END_TOKEN}" in rendered
    assert '<|call:tool|>calculator({"expr": "12*12"})<|return:tool|>' in rendered
    assert '<|start_header_id|>tool<|end_header_id|>\n\n<|return:tool|>{"result": 144}<|end_of_text|>' in rendered
    assert rendered.endswith(f"<|start_header_id|>assistant<|end_header_id|>\n\n{THOUGHT_START_TOKEN}\n")

    token_ids = tok.apply_chat_template(
        messages=messages,
        tools=tools,
        hma_context="Working Memory: Active session initialized.",
        epistemic_guard=True,
        add_generation_prompt=True,
        enable_thinking=True,
        tokenize=True,
    )
    assert isinstance(token_ids, list)
    assert tok.special_tokens[IMAGE_TOKEN] in token_ids
    assert tok.special_tokens[MEMORY_START_TOKEN] in token_ids
    assert tok.special_tokens[THOUGHT_START_TOKEN] in token_ids


@pytest.mark.unit
def test_apply_chat_template_refusal_turn() -> None:
    """Verify explicit epistemic refusal turn rendering."""
    tok = ByteLevelBPETokenizer()
    messages = [
        {"role": "user", "content": "What is xylophia-99?"},
        {
            "role": "assistant",
            "refusal": "This topic is not yet in my current training session.",
        },
    ]
    rendered = tok.apply_chat_template(messages=messages, tokenize=False)
    assert "<|refusal|>This topic is not yet in my current training session.<|end_of_text|>" in rendered
