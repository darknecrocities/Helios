"""Rich Console UI, Typewriter Animation, and Visual Components for Helios Chat.

Provides glowing branded header panels, animated typewriter streaming,
collapsible Chain-of-Thought (CoT) panels, models comparison tables,
and status footers for an elite developer terminal experience.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_typewriter_stream(text: str, delay: float = 0.007) -> None:
    """Print text token-by-token with organic typewriter cadence."""
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def render_chat_banner(tier_info: dict[str, Any], device: Any) -> None:
    """Render glowing cyan/gold header banner in a stylish Rich panel."""
    content = (
        f"Helios Foundation Interactive Terminal Chat (v0.3.0)\n"
        f"[bold yellow]☀️  HELIOS COGNITIVE RUNTIME[/bold yellow]\n"
        f"[dim]Lead Architect & AI Engineer: Arron Kian Parejas <parejasarronkian@gmail.com>[/dim]\n\n"
        f"  [bold cyan]● Active Model  :[/bold cyan] [bold white]{tier_info['display_name']} ({tier_info['version']})[/bold white]\n"
        f"  [bold cyan]● Parameters    :[/bold cyan] [white]{tier_info['params']} tied parameters[/white]\n"
        f"  [bold cyan]● Context Window:[/bold cyan] [white]{tier_info['context']} (RoPE θ={tier_info['rope_theta']})[/white]\n"
        f"  [bold cyan]● Hardware Tier :[/bold cyan] [white]{device} · {tier_info['ram_target']}[/white]\n"
        f"  [bold cyan]● Memory Store  :[/bold cyan] [white]HMA SQLite WAL + In-Memory Semantic BM25[/white]\n"
        f"  [bold cyan]● External Disk :[/bold cyan] [white]/Volumes/One Touch/Helios/ (0B internal SSD)[/white]\n\n"
        f"[dim cyan]Commands: /models · /model <tier> · /think <on|off> · /memory · /stats · /reset · /exit[/dim cyan]"
    )
    console.print(
        Panel(
            content,
            border_style="cyan",
            title="[bold yellow]⚡ HELIOS AI COGNITIVE RUNTIME[/bold yellow]",
            subtitle="[dim]Hierarchical Memory Architecture + Dynamic CoT[/dim]",
        )
    )


def render_thought_panel(thought_text: str) -> None:
    """Render the Chain-of-Thought reasoning block in a subtle cyan panel."""
    if not thought_text.strip():
        return
    console.print(
        Panel(
            f"[dim italic]{thought_text.strip()}[/dim italic]",
            border_style="dim cyan",
            title="[bold cyan]💭 Helios Chain-of-Thought Reasoning[/bold cyan]",
            subtitle="[dim]Step-by-step first-principles deduction[/dim]",
        )
    )


def render_models_table(current_preset: str, catalog: dict[str, dict[str, str]]) -> None:
    """Render a clean, formatted comparison table of all 6 Helios model tiers."""
    table = Table(
        title="Helios Foundation Model Family Catalog",
        border_style="cyan",
        header_style="bold yellow",
    )
    table.add_column("Status", justify="center", width=8)
    table.add_column("Tier Name", style="bold white", width=16)
    table.add_column("Preset", style="cyan", width=12)
    table.add_column("Parameters", justify="right", width=14)
    table.add_column("Context", justify="right", width=12)
    table.add_column("Q4 Size", justify="right", width=10)
    table.add_column("Target Hardware & Memory Ceiling", style="dim", width=42)

    for key, info in catalog.items():
        is_active = (current_preset == info["preset"]) or (current_preset == key)
        status_badge = "[bold green]● ACTIVE[/bold green]" if is_active else "[dim]○[/dim]"
        table.add_row(
            status_badge,
            info["display_name"],
            info["preset"],
            info["params"],
            info["context"].split()[0],
            info["q4_size"],
            info["ram_target"],
        )
    console.print(table)


def render_model_switch_card(tier_info: dict[str, str]) -> None:
    """Display confirmation card when switching active model tier live."""
    content = (
        f"[bold green]✓ Switched Active Model to {tier_info['display_name']}[/bold green]\n\n"
        f"  [cyan]Preset Identifier:[/cyan] [bold white]{tier_info['preset']}[/bold white]\n"
        f"  [cyan]Real Parameters  :[/cyan] [white]{tier_info['params']}[/white]\n"
        f"  [cyan]Context Window   :[/cyan] [white]{tier_info['context']}[/white]\n"
        f"  [cyan]RoPE Base Theta  :[/cyan] [white]{tier_info['rope_theta']}[/white]\n"
        f"  [cyan]GGUF Q4 Footprint:[/cyan] [white]{tier_info['q4_size']}[/white]\n"
        f"  [cyan]Hardware Target  :[/cyan] [dim]{tier_info['ram_target']}[/dim]"
    )
    console.print(Panel(content, border_style="green", title="[bold]Active Tier Switched[/bold]"))
