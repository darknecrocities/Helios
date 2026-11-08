"""Adversarial Stress Test Suite for Milestone 7 (Features 38 and 39).

Targets:
1. CLI case-insensitivity: bizarre casing combinations, trailing options with
   uppercase subcommands, and help flag permutations.
2. Slash command parsing: extreme whitespace, unicode whitespace, extra arguments,
   malformed slash inputs, injection attempts, empty inputs, and unknown commands.
3. Verification that NO slash command ever falls through to LLM neural generation.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import random
import string
from unittest.mock import MagicMock

import pytest
import torch
from typer.testing import CliRunner

from helios.cli.chat import ChatSession
from helios.cli.commands import (
    CommandResult,
    dispatch_chat_command,
    parse_slash_command,
)
from helios.cli.main import app
from helios.cli.utils import load_model_and_config, load_or_create_tokenizer

runner = CliRunner()


@pytest.fixture(scope="module")
def shared_session(tmp_path_factory: pytest.TempPathFactory) -> ChatSession:
    """Provide a shared lightweight ChatSession for fast dispatch testing."""
    tmp_dir = tmp_path_factory.mktemp("adversarial_chat")
    model, config = load_model_and_config(preset="nano-15m", device="cpu")
    tokenizer = load_or_create_tokenizer(vocab_size=config.vocab_size)
    return ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id="adv_test_session",
        db_path=tmp_dir / "adv_chat.db",
        device=torch.device("cpu"),
        preset="nano-15m",
    )


# ==============================================================================
# 1. CLI Case-Insensitivity & Trailing Options Adversarial Tests (Feature 38)
# ==============================================================================


@pytest.mark.tier5
class TestCLICaseInsensitivityAdversarial:
    """Adversarially probe CLI case-insensitivity across bizarre casing and options."""

    @pytest.mark.parametrize(
        "subcommand",
        [
            "cHaT",
            "GeNeRaTe",
            "EvAl",
            "InSpEcT",
            "sMoKe_TeSt",
            "rEsUmE_tEsT",
            "AuTo_ImPrOvE",
            "ExPoRt",
            "VeRsIoN",
            "RoAdMaP",
        ],
    )
    def test_bizarre_casing_all_subcommands_help(self, subcommand: str) -> None:
        """Verify that every subcommand accepts bizarre casing with --help."""
        result = runner.invoke(app, [subcommand, "--help"])
        assert result.exit_code == 0, f"Failed on {subcommand}: {result.output}"
        assert "Usage: " in result.stdout

    @pytest.mark.parametrize(
        ("group", "subcommand"),
        [
            ("eXpOrT", "hF"),
            ("ExPoRt", "OLLAMA"),
            ("eXpOrT", "DiSk"),
            ("ExPoRt", "PhYsIcAl"),
        ],
    )
    def test_bizarre_casing_nested_export(self, group: str, subcommand: str) -> None:
        """Verify that nested export subcommands accept bizarre casing."""
        result = runner.invoke(app, [group, subcommand, "--help"])
        assert result.exit_code == 0, f"Failed on {group} {subcommand}: {result.output}"
        assert "Usage: " in result.stdout

    def test_inspect_trailing_options_casing(self) -> None:
        """Verify inspect command executes with bizarre casing and trailing options."""
        res_short = runner.invoke(app, ["InSpEcT", "-m", "nano-20m"])
        assert res_short.exit_code == 0
        assert "Helios Model Inspector" in res_short.stdout
        assert "nano-20m" in res_short.stdout

        res_long = runner.invoke(app, ["iNsPeCt", "--model-size", "nano-25m"])
        assert res_long.exit_code == 0
        assert "Helios Model Inspector" in res_long.stdout
        assert "nano-25m" in res_long.stdout

    def test_smoke_test_trailing_options_casing(self) -> None:
        """Verify smoke_test executes convergence with mixed casing and options."""
        res = runner.invoke(app, ["sMoKe_TeSt", "--device", "cpu", "--max-steps", "150"])
        assert res.exit_code == 0
        assert "Convergence Success" in res.stdout

    def test_generate_trailing_options_casing(self) -> None:
        """Verify generate command executes with bizarre casing and full arguments."""
        res = runner.invoke(
            app,
            [
                "GeNeRaTe",
                "--prompt",
                "def add(a, b):",
                "--max-new-tokens",
                "2",
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
                "--no-stream",
            ],
        )
        assert res.exit_code == 0
        assert "Helios Generation Telemetry" in res.stdout

    def test_eval_trailing_options_casing(self) -> None:
        """Verify eval command executes with bizarre casing and benchmark options."""
        res = runner.invoke(
            app,
            [
                "EvAl",
                "--preset",
                "nano-15m",
                "--device",
                "cpu",
                "--prompt-len",
                "2",
                "--gen-len",
                "2",
            ],
        )
        assert res.exit_code == 0
        assert "Helios Performance & Perplexity Report" in res.stdout

    def test_chat_trailing_options_casing(self) -> None:
        """Verify chat command initializes with bizarre casing and exits cleanly."""
        res = runner.invoke(
            app,
            ["cHaT", "--preset", "nano-15m", "--device", "cpu"],
            input="/exit\n",
        )
        assert res.exit_code == 0
        assert "Exiting Helios Chat. Goodbye!" in res.stdout

    def test_resume_test_bizarre_casing(self) -> None:
        """Verify resume_test executes with bizarre casing."""
        res = runner.invoke(app, ["rEsUmE_tEsT"])
        assert res.exit_code == 0
        assert "Resumption Equivalence Passed" in res.stdout

    def test_genuine_unknown_command_exits_code_2(self) -> None:
        """Verify genuine unknown command fails with code 2."""
        res = runner.invoke(app, ["bIzArRe_FaKe_CmD"])
        assert res.exit_code == 2
        assert "No such command" in res.output


# ==============================================================================
# 2. Slash Command Parsing Adversarial Probing (Feature 39)
# ==============================================================================


@pytest.mark.tier5
class TestSlashCommandParsingAdversarial:
    """Adversarially probe in-chat slash command parsing and argument handling."""

    def test_extreme_whitespace_parsing(self, shared_session: ChatSession) -> None:
        """Verify commands with extreme tabs, newlines, and whitespace normalize cleanly."""
        parsed = parse_slash_command("/think   \t\n  on")
        assert parsed == ("/think", "on")

        parsed_tab = parse_slash_command("  \t\n  /switch   \t\n  alpha  \t\n  ")
        assert parsed_tab == ("/switch", "alpha")

        res_on = dispatch_chat_command("/think   \t\n  on", shared_session)
        assert res_on.handled is True
        assert shared_session.show_thought is True

        res_off = dispatch_chat_command("/think \t\t off \n\n", shared_session)
        assert res_off.handled is True
        assert shared_session.show_thought is False

    def test_unicode_whitespace_handling(self, shared_session: ChatSession) -> None:
        """Verify commands with non-breaking and ideographic spaces parse correctly."""
        # Non-breaking space \u00A0
        parsed_nbsp = parse_slash_command("\u00a0/think\u00a0on\u00a0")
        assert parsed_nbsp is not None
        assert parsed_nbsp[0] == "/think"
        res_nbsp = dispatch_chat_command("\u00a0/think\u00a0on\u00a0", shared_session)
        assert res_nbsp.handled is True
        assert shared_session.show_thought is True

        # Ideographic space \u3000
        parsed_ideo = parse_slash_command("\u3000/models\u3000")
        assert parsed_ideo == ("/models", "")
        res_ideo = dispatch_chat_command("\u3000/models\u3000", shared_session)
        assert res_ideo.handled is True

        # Em space \u2003
        parsed_em = parse_slash_command("\u2003/help\u2003")
        assert parsed_em == ("/help", "")
        res_em = dispatch_chat_command("\u2003/help\u2003", shared_session)
        assert res_em.handled is True

    @pytest.mark.parametrize(
        ("cmd_str", "expected_exit"),
        [
            ("/exit now please", True),
            ("/quit 0 gracefully", True),
            ("/models all available tiers", False),
            ("/reset immediately", False),
            ("/clear conversation history", False),
            ("/memory verbose diagnostic", False),
            ("/stats extended", False),
            ("/help commands and syntax", False),
            ("/think on please", False),
            ("/think off thanks", False),
            ("/model flagship extra arguments", False),
            ("/switch alpha trailing args", False),
        ],
    )
    def test_commands_with_extra_arguments(
        self, shared_session: ChatSession, cmd_str: str, expected_exit: bool
    ) -> None:
        """Verify commands with trailing arguments execute safely without unhandled errors."""
        res = dispatch_chat_command(cmd_str, shared_session)
        assert res.handled is True
        assert res.should_exit is expected_exit

    @pytest.mark.parametrize(
        "malformed_cmd",
        [
            "///help",
            "/??",
            "//",
            "/",
            "//models",
            "///",
            "/!",
            "/@#$%",
            "/...",
            "/exit/",
            "/???",
        ],
    )
    def test_malformed_slash_inputs_intercepted(
        self, shared_session: ChatSession, malformed_cmd: str
    ) -> None:
        """Verify that malformed slash patterns are intercepted as unknown commands."""
        res = dispatch_chat_command(malformed_cmd, shared_session)
        assert res.handled is True
        assert res.should_exit is False

    @pytest.mark.parametrize(
        "injection_input",
        [
            "/model; rm -rf /",
            "/switch $(whoami)",
            "/think ' or '1'='1",
            "/model <script>alert(1)</script>",
            "/model \x00\x00",
            "/model $(cat /etc/passwd)",
            "/__import__('os').system('id')",
            "/eval print('pwned')",
            "/exit; reboot",
        ],
    )
    def test_injection_attempts_neutralized(
        self, shared_session: ChatSession, injection_input: str
    ) -> None:
        """Verify injection strings in commands are neutralized and never executed."""
        res = dispatch_chat_command(injection_input, shared_session)
        assert res.handled is True

    @pytest.mark.parametrize(
        "empty_or_ws",
        [
            "",
            "   ",
            "\t\n\r",
            "\u00a0\u3000",
        ],
    )
    def test_empty_and_whitespace_inputs(self, empty_or_ws: str) -> None:
        """Verify empty or pure whitespace inputs return None from parser."""
        parsed = parse_slash_command(empty_or_ws)
        assert parsed is None

    @pytest.mark.parametrize(
        "unknown_cmd",
        [
            "/unknown",
            "/notacommand",
            "/foobar",
            "/modle",
            "/switc",
            "/modelx",
            "/thinkfast",
            "/heeeeelp",
        ],
    )
    def test_unknown_commands_intercepted(
        self, shared_session: ChatSession, unknown_cmd: str
    ) -> None:
        """Verify typos and unknown commands are intercepted without LLM generation."""
        res = dispatch_chat_command(unknown_cmd, shared_session)
        assert res.handled is True
        assert res.should_exit is False


# ==============================================================================
# 3. Verification of Zero Fallthrough to LLM Neural Generation (Invariant)
# ==============================================================================


@pytest.mark.tier5
class TestSlashCommandZeroFallthrough:
    """Verify that NO slash command ever falls through to LLM neural generation."""

    def test_zero_fallthrough_on_extensive_slash_corpus(self, shared_session: ChatSession) -> None:
        """Exhaustively verify 200+ malformed/fuzzed slash commands never call handle_turn."""
        mock_turn = MagicMock()
        mock_turn.side_effect = AssertionError("Neural generation triggered for slash command!")
        shared_session.handle_turn = mock_turn

        base_corpus = [
            "/",
            "//",
            "///",
            "////",
            "///help",
            "/??",
            "/?",
            "/!",
            "/@#$%",
            "/models",
            "/models all",
            "/model",
            "/model ; rm -rf /",
            "/switch",
            "/switch invalid_xyz",
            "/think",
            "/think on",
            "/think off",
            "/think invalid_arg",
            "/memory",
            "/stats",
            "/reset",
            "/clear",
            "/help",
            "/unknown",
            "/not_a_cmd",
            "/../etc/passwd",
            "/\\x00",
            "/12345",
            "/🤖",
        ]

        chars = string.ascii_letters + string.punctuation + string.digits
        fuzz_corpus = [
            "/" + "".join(random.choices(chars, k=random.randint(1, 40))) for _ in range(150)
        ]

        full_corpus = base_corpus + fuzz_corpus

        for cmd_input in full_corpus:
            res = dispatch_chat_command(cmd_input, shared_session)
            assert res.handled is True, f"Command '{cmd_input}' fell through (handled=False)!"
            # Confirm mock_turn was never called
            mock_turn.assert_not_called()

    def test_conversational_input_properly_passes_through(
        self, shared_session: ChatSession
    ) -> None:
        """Verify that genuine non-slash conversational text returns handled=False."""
        conversational_queries = [
            "What is the capital of the Philippines?",
            "Explain quantum entanglement.",
            "Write a Python function to compute Fibonacci numbers.",
            "Who is Arron Kian Parejas?",
            "Solve 42 * 17",
            "1 + 1",
            "Is /models a command?",
        ]

        for query in conversational_queries:
            res = dispatch_chat_command(query, shared_session)
            assert res == CommandResult(handled=False, should_exit=False), (
                f"Query '{query}' was incorrectly intercepted as a command!"
            )

    def test_interactive_chat_runner_adversarial_stream(self) -> None:
        """Verify interactive chat runner handles multi-slash adversarial stream cleanly."""
        stream_inputs = (
            "\n".join(
                [
                    "/models",
                    "/models all",
                    "/think",
                    "/think on",
                    "/think   \t\n  off",
                    "/memory",
                    "/stats",
                    "/help",
                    "///help",
                    "/??",
                    "/model; rm -rf /",
                    "/notacommand",
                    "/reset",
                    "/exit",
                ]
            )
            + "\n"
        )

        res = runner.invoke(
            app,
            ["cHaT", "--preset", "nano-15m", "--device", "cpu"],
            input=stream_inputs,
        )
        assert res.exit_code == 0, f"Chat exited with error: {res.output}"
        assert "Exiting Helios Chat. Goodbye!" in res.stdout
        assert "Unrecognized command" in res.stdout
        assert "Chain-of-Thought" in res.stdout
        assert "Session working memory context cleared." in res.stdout
