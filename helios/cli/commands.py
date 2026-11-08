"""Helios Interactive Chat Slash Commands Tokenization and Routing.

Provides hardened command parsing, whitespace normalization, case-insensitive
argument extraction, and instantaneous execution of all in-chat commands.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.panel import Panel

from helios.cli.ui import (
    console,
    render_model_switch_card,
    render_models_table,
)
from helios.cli.utils import (
    MODEL_FAMILY_CATALOG,
    resolve_tier_key,
)

if TYPE_CHECKING:
    from helios.cli.chat import ChatSession

VALID_SLASH_COMMANDS: frozenset[str] = frozenset(
    {
        "/models",
        "/model",
        "/switch",
        "/think",
        "/memory",
        "/stats",
        "/reset",
        "/clear",
        "/help",
        "/exit",
        "/quit",
    }
)

CHAT_HELP_TEXT: str = (
    "Special Chat Commands:\n"
    "  /models          : List all 6 Helios models with parameters, context, and memory specs\n"
    "  /model <tier>    : Switch active model live (e.g. /model alpha, /model flagship)\n"
    "  /switch <tier>   : Alias for /model to switch active model live\n"
    "  /think <on|off>  : Enable/disable Chain-of-Thought panel; /think displays current state\n"
    "  /memory          : Display HMA memory state and SQLite persistence details\n"
    "  /stats           : Display session generation performance and throughput\n"
    "  /reset, /clear   : Clear working memory context and turn history\n"
    "  /help            : Display this chat command reference panel\n"
    "  /exit, /quit     : Terminate chat session"
)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of slash command evaluation."""

    handled: bool
    should_exit: bool = False


def parse_slash_command(raw_input: str) -> tuple[str, str] | None:
    """Tokenize and normalize an interactive chat slash command.

    Strips leading and trailing whitespace, extracts the command token in lowercase,
    and returns the normalized argument string.

    Returns:
        tuple[str, str]: (normalized_cmd, normalized_arg) if raw_input starts with '/',
        None if raw_input is not a slash command.
    """
    cleaned = raw_input.strip()
    if not cleaned.startswith("/"):
        return None

    parts = cleaned.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg


def handle_models(session: ChatSession, arg: str = "") -> None:
    """Render the model family catalog table."""
    render_models_table(session.preset, MODEL_FAMILY_CATALOG)


def handle_model_switch(session: ChatSession, arg: str) -> None:
    """Switch active model tier with whitespace normalization and case insensitivity."""
    normalized_arg = " ".join(arg.split())
    if not normalized_arg:
        render_models_table(session.preset, MODEL_FAMILY_CATALOG)
        console.print(
            "[dim cyan]Usage: /model <tier> or /switch <tier> (e.g. /model alpha, /switch flagship)[/dim cyan]"
        )
        return

    try:
        tier_k = resolve_tier_key(normalized_arg)
        session.switch_model(normalized_arg)
        render_model_switch_card(MODEL_FAMILY_CATALOG[tier_k])
    except (KeyError, ValueError) as exc:
        msg = exc.args[0] if exc.args else str(exc)
        console.print(f"[bold red]✗ {msg}[/bold red]")
        render_models_table(session.preset, MODEL_FAMILY_CATALOG)
    except (MemoryError, RuntimeError) as exc:
        console.print(f"[bold yellow]⚠️  Model switch safeguard engaged: {exc}[/bold yellow]")
    except Exception as exc:
        console.print(f"[bold red]✗ Model switch failed: {exc}[/bold red]")
        render_models_table(session.preset, MODEL_FAMILY_CATALOG)


def handle_think(session: ChatSession, arg: str) -> None:
    """Configure or display Chain-of-Thought reasoning panel."""
    normalized_arg = arg.strip().lower()
    if not normalized_arg:
        st = "enabled" if session.show_thought else "disabled"
        console.print(
            f"[bold cyan]ℹ Chain-of-Thought (CoT) display is currently {st}. "
            "Use '/think on' or '/think off' to toggle.[/bold cyan]"
        )
        return

    if normalized_arg in ("on", "1", "true", "enable", "enabled"):
        session.show_thought = True
        console.print("[bold cyan]ℹ Chain-of-Thought (CoT) display enabled.[/bold cyan]")
    elif normalized_arg in ("off", "0", "false", "disable", "disabled"):
        session.show_thought = False
        console.print("[bold cyan]ℹ Chain-of-Thought (CoT) display disabled.[/bold cyan]")
    elif normalized_arg in ("toggle",):
        session.show_thought = not session.show_thought
        st = "enabled" if session.show_thought else "disabled"
        console.print(f"[bold cyan]ℹ Chain-of-Thought (CoT) display {st}.[/bold cyan]")
    else:
        console.print(
            f"[bold red]✗ Invalid argument '{arg}' for /think. Usage: /think <on|off>[/bold red]"
        )


def handle_memory(session: ChatSession, arg: str = "") -> None:
    """Display HMA memory status."""
    console.print(session.memory_status())


def handle_stats(session: ChatSession, arg: str = "") -> None:
    """Display session generation metrics."""
    console.print(session.stats_status())


def handle_reset(session: ChatSession, arg: str = "") -> None:
    """Clear session working memory context."""
    session.clear()
    console.print("[bold green]✓ Session working memory context cleared.[/bold green]")


def handle_help() -> None:
    """Print available chat slash commands."""
    console.print(
        Panel(CHAT_HELP_TEXT, border_style="cyan", title="[bold]Chat Command Reference[/bold]")
    )


def handle_unknown(cmd: str) -> None:
    """Intercept unknown slash commands with helpful syntax error guidance."""
    console.print(
        f"[bold red]✗ Unrecognized command '{cmd}'. Type /help for available commands.[/bold red]"
    )


def dispatch_chat_command(user_input: str, session: ChatSession) -> CommandResult:
    """Route user input if it is a slash command.

    Returns:
        CommandResult with handled=True if the input was processed as a command,
        or handled=False if it should be sent to the model/LLM for conversation.
    """
    parsed = parse_slash_command(user_input)
    if parsed is None:
        return CommandResult(handled=False, should_exit=False)

    cmd, arg = parsed

    if cmd in ("/exit", "/quit"):
        console.print("[yellow]Exiting Helios Chat. Goodbye![/yellow]")
        return CommandResult(handled=True, should_exit=True)

    if cmd in ("/reset", "/clear"):
        handle_reset(session, arg)
    elif cmd == "/models":
        handle_models(session, arg)
    elif cmd in ("/model", "/switch"):
        handle_model_switch(session, arg)
    elif cmd == "/think":
        handle_think(session, arg)
    elif cmd == "/memory":
        handle_memory(session, arg)
    elif cmd == "/stats":
        handle_stats(session, arg)
    elif cmd == "/help":
        handle_help()
    else:
        handle_unknown(cmd)

    return CommandResult(handled=True, should_exit=False)
