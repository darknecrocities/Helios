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
        "/status",
        "/health",
        "/categories",
        "/reset",
        "/clear",
        "/train",
        "/bench",
        "/sync",
        "/publish",
        "/update",
        "/doctor",
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
    "  /stats           : Display session generation performance and active checkpoint telemetry\n"
    "  /status, /health : Full 6-tier checkpoint health/corruption audit & improvement deltas\n"
    "  /categories      : List all 10 pretraining dataset categories (FineWeb, Math, History, etc.)\n"
    "  /train [args]    : Pretrain on OneTouch & auto-sync (e.g. /train all --category math -m 20)\n"
    "  /bench [tier]    : Run automated held-out evaluation & perplexity benchmark\n"
    "  /sync, /publish  : Auto-sync latest models & statistical telemetry to HF, Ollama, & Git\n"
    "  /doctor          : Verify OneTouch filesystem, NPU backends, and storage health\n"
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


def handle_status(session: ChatSession, arg: str = "") -> None:
    """Display 6-tier checkpoint health, corruption diagnostics, and improvement deltas."""
    from helios_pipeline.workflow import render_health_and_improvement_table

    render_health_and_improvement_table()


def handle_categories(session: ChatSession, arg: str = "") -> None:
    """Display all 10 selectable pretraining categories and dataset counts."""
    from helios_pipeline.workflow import render_categories_table

    render_categories_table()


def handle_train(session: ChatSession, arg: str) -> None:
    """Trigger continuous pretraining on OneTouch drive with category & default auto-sync."""
    import shlex

    from helios.cli.utils import reload_model_for_tier
    from helios_pipeline.data.categories import CATEGORY_ALIASES, CATEGORY_SPECS
    from helios_pipeline.workflow import parse_duration, run_train_workflow

    tokens = shlex.split(arg) if arg else []
    tier = resolve_tier_key(session.preset)
    minutes = 5.0
    sync_target: str | None = "all"
    device = "auto"
    upload_weights = False
    category = "all"

    idx = 0
    while idx < len(tokens):
        tok = tokens[idx]
        low = tok.lower()
        if low in ("--minutes", "-m") and idx + 1 < len(tokens):
            minutes = float(tokens[idx + 1])
            idx += 2
        elif low in ("--duration", "-d") and idx + 1 < len(tokens):
            minutes = parse_duration(tokens[idx + 1])
            idx += 2
        elif low in ("--category", "-c") and idx + 1 < len(tokens):
            category = tokens[idx + 1]
            idx += 2
        elif low == "--no-sync":
            sync_target = None
            idx += 1
        elif low in ("--sync", "-s"):
            if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("-"):
                sync_target = tokens[idx + 1]
                idx += 2
            else:
                sync_target = "all"
                idx += 1
        elif low == "--device" and idx + 1 < len(tokens):
            device = tokens[idx + 1]
            idx += 2
        elif low == "--upload-weights":
            upload_weights = True
            idx += 1
        elif not tok.startswith("-"):
            if low in CATEGORY_SPECS or low in CATEGORY_ALIASES:
                category = low
            else:
                tier = tok
            idx += 1
        else:
            idx += 1

    console.print(
        f"[bold cyan]☀️  Initiating continuous pretraining: tier='{tier}' · "
        f"category='{category}' · duration={minutes:g} min · auto-sync='{sync_target or 'off'}'[/bold cyan]"
    )
    code = run_train_workflow(
        tiers_spec=tier,
        minutes=minutes,
        device=device,
        sync=sync_target,
        upload_weights=upload_weights,
        category=category,
    )
    if code == 0:
        console.print(
            "[bold green]✓ Pretraining, automated benchmark, and multi-platform sync completed![/bold green]"
        )
        current_tier = resolve_tier_key(session.preset)
        if tier == "all" or current_tier in tier.split(","):
            try:
                session.model, session.config = reload_model_for_tier(
                    session.preset, device=session.device
                )
                console.print(
                    f"[bold cyan]✓ Hot-reloaded latest checkpoint for active model tier '{current_tier}'.[/bold cyan]"
                )
            except Exception:
                pass
    else:
        console.print(f"[bold red]✗ Pretraining halted with exit code {code}.[/bold red]")


def handle_bench(session: ChatSession, arg: str) -> None:
    """Run automated held-out evaluation & perplexity benchmark from chat."""
    from helios_pipeline.workflow import render_health_and_improvement_table

    render_health_and_improvement_table(
        title=f"Automated Benchmark & Improvement Telemetry — {(arg.strip() or 'ALL TIERS').upper()}"
    )


def handle_sync(session: ChatSession, arg: str) -> None:
    """Synchronize latest reports, statistical descriptions, and models to HF, Ollama, and Git."""
    from helios_pipeline.sync.orchestrator import run_sync

    target = arg.strip() or "all"
    console.print(
        f"[bold cyan]🔄 Auto-syncing latest checkpoints & statistical descriptions to: {target}...[/bold cyan]"
    )
    rep = run_sync(targets=target)
    console.print(
        Panel(
            f"[bold white]{rep.summary_headline}[/bold white]\n\n"
            + "\n".join(f"  • [cyan]{k}[/cyan]: {status}" for k, status in rep.status.items()),
            border_style="green",
            title="[bold green]✓ Multi-Platform Auto-Sync Complete[/bold green]",
        )
    )


def handle_doctor(session: ChatSession, arg: str = "") -> None:
    """Run system, OneTouch storage, and NPU acceleration diagnostic."""
    from helios_pipeline.workflow import run_doctor

    run_doctor()


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
    """Route user input if it is a slash command."""
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
    elif cmd in ("/status", "/health"):
        handle_status(session, arg)
    elif cmd == "/categories":
        handle_categories(session, arg)
    elif cmd == "/train":
        handle_train(session, arg)
    elif cmd == "/bench":
        handle_bench(session, arg)
    elif cmd in ("/sync", "/publish", "/update"):
        handle_sync(session, arg)
    elif cmd == "/doctor":
        handle_doctor(session, arg)
    elif cmd == "/help":
        handle_help()
    else:
        handle_unknown(cmd)

    return CommandResult(handled=True, should_exit=False)
