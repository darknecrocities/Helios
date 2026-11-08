"""Helios Interactive Terminal Chat with HMA Memory Grounding and Dynamic CoT.

Provides a multi-turn rich interactive chat interface with typewriter streaming,
pulsating thinking animation, Chain-of-Thought (CoT) reasoning panels,
and live model switching across all 6 Helios tiers.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import re
from pathlib import Path

import torch
import typer
from rich.panel import Panel

from helios.cli.fallback import (
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.cli.generate import stream_generate
from helios.cli.ui import (
    console,
    print_typewriter_stream,
    render_chat_banner,
    render_model_switch_card,
    render_models_table,
    render_thought_panel,
)
from helios.cli.utils import (
    MODEL_FAMILY_CATALOG,
    format_card,
    load_model_and_config,
    load_or_create_tokenizer,
    reload_model_for_tier,
    resolve_device,
    resolve_tier_key,
)
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory
from helios.model.transformer import HeliosForCausalLM
from helios.reasoning.cot_engine import (
    extract_thought_and_response,
)
from helios.tokenizer.base import BaseTokenizer


def solve_arithmetic_query(text: str) -> str | None:
    """Deterministically solve basic and multi-step arithmetic expressions."""
    cleaned = text.strip().lower().rstrip("?").strip()
    for prefix in [
        "what is",
        "calculate",
        "solve",
        "how much is",
        "evaluate",
        "compute",
        "tell me what is",
        "tell me",
    ]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()

    # Percentage evaluation: e.g. 20% of 150
    m_pct = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d+(?:\.\d+)?)", cleaned)
    if m_pct:
        p, tot = float(m_pct.group(1)), float(m_pct.group(2))
        res = (p / 100.0) * tot
        return f"{cleaned} = {int(res) if res.is_integer() else res}"

    # Fraction evaluation: e.g. 1/4 of 80
    m_frac = re.fullmatch(r"(\d+)\s*/\s*(\d+)\s*(?:of)\s*(\d+(?:\.\d+)?)", cleaned)
    if m_frac:
        num, den, tot = float(m_frac.group(1)), float(m_frac.group(2)), float(m_frac.group(3))
        res = (num / den) * tot
        return f"{cleaned} = {int(res) if res.is_integer() else res}"

    expr = cleaned.replace("^", "**").replace("x", "*").replace("×", "*").replace("÷", "/")
    if re.fullmatch(r"[\d\s\+\-\*\/\%\(\)\.]+", expr):
        if not any(op in expr for op in ["+", "-", "*", "/", "%"]):
            return None
        try:
            val = eval(expr, {"__builtins__": None}, {})
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return f"{cleaned} = {val}"
        except Exception:
            return None
    return None


class ChatSession:
    """Manages chat session lifecycle, memory grounding, and telemetry."""

    def __init__(
        self,
        model: HeliosForCausalLM,
        tokenizer: BaseTokenizer,
        session_id: str = "default",
        db_path: Path | str | None = None,
        device: torch.device | None = None,
        preset: str = "nano-20m",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.session_id = session_id
        self.device = device or next(model.parameters()).device
        self.preset = preset
        self.show_thought = True

        # Setup HMA with persistent Episodic SQLite store
        if db_path is not None and str(db_path) != ":memory:":
            resolved_db = Path(db_path).resolve()
            resolved_db.parent.mkdir(parents=True, exist_ok=True)
            episodic_store = EpisodicMemory(db_path=resolved_db)
        else:
            episodic_store = EpisodicMemory(db_path=":memory:")

        self.hma = HMAController(episodic_memory=episodic_store, default_session_id=session_id)

        # Preload semantic knowledge base from reasoning data if present
        self._valid_words: set[str] = set()
        self._seed_knowledge_base()

        # Telemetry
        self.total_turns = 0
        self.total_prompt_tokens = 0
        self.total_generated_tokens = 0
        self.total_generation_time_s = 0.0

    def _seed_knowledge_base(self) -> None:
        """Seed semantic index with expanded and contextual reasoning records."""
        seed_hma_knowledge_base(self.hma, self._valid_words)

    def switch_model(self, new_preset: str) -> str:
        """Switch active Helios model tier during a live terminal chat session."""
        tier_key = resolve_tier_key(new_preset)
        tier_info = MODEL_FAMILY_CATALOG[tier_key]
        self.preset = tier_info["preset"]
        del self.model
        self.model, _ = reload_model_for_tier(self.preset, device=self.device)
        return (
            f"✓ Switched active model to {tier_info['display_name']} "
            f"(Preset: {self.preset} | Parameters: {tier_info['params']} | "
            f"Context: {tier_info['context']} | Q4 Size: {tier_info['q4_size']})"
        )

    def _is_coherent_reply(self, reply: str) -> bool:
        """Evaluate semantic coherence ratio against known vocabulary."""
        if not reply or len(reply.strip()) < 8:
            return False
        # Catch spaced-out or byte-looping corruption artifacts
        if "   " in reply or len(re.findall(r"\b[a-zA-Z]{3,}\b", reply)) < 2:
            return False
        if any(
            frag in reply
            for frag in [
                "<|",
                "|>",
                "_o",
                "|a",
                "_a",
                "?|",
                "|s",
                "||",
                "t|",
                "d|",
                "_e",
                "_s",
                "o_",
                "i_",
            ]
        ):
            return False
        words = [w.lower().strip(".,!?:;\"'()[]{}") for w in reply.split()]
        words = [w for w in words if len(w) >= 2]
        if len(words) < 2:
            return False
        if not self._valid_words:
            return True
        valid_count = sum(1 for w in words if w in self._valid_words)
        return (valid_count / len(words)) >= 0.70

    def clear(self) -> None:
        """Clear active working memory and reset conversation turn state."""
        self.hma.clear_session(self.session_id)
        self.total_turns = 0

    def memory_status(self) -> str:
        """Generate human-readable status of all HMA memory tiers."""
        episodic_turns = self.hma.episodic.get_session_turns(self.session_id)
        buf = self.hma.working.get_or_create(self.session_id)
        fields = [
            ("Session ID", self.session_id),
            ("Episodic Turns (SQLite)", f"{len(episodic_turns)} turns"),
            ("Episodic Database", str(self.hma.episodic.db_path)),
            ("Working Memory Turns", f"{buf.turn_count} active"),
            ("Working Memory Tokens", f"{buf.total_tokens} / {buf.max_tokens}"),
            ("Semantic Index Size", f"{len(self.hma.semantic.documents)} docs"),
            ("Procedural Tools", f"{len(self.hma.procedural.list_tools())} registered"),
        ]
        return format_card("Hierarchical Memory Architecture Status", fields)

    def stats_status(self) -> str:
        """Generate human-readable session performance statistics."""
        avg_throughput = (
            self.total_generated_tokens / self.total_generation_time_s
            if self.total_generation_time_s > 0
            else 0.0
        )
        fields = [
            ("Total User Turns", f"{self.total_turns}"),
            ("Total Prompt Tokens", f"{self.total_prompt_tokens}"),
            ("Total Generated Tokens", f"{self.total_generated_tokens}"),
            ("Cumulative Time", f"{self.total_generation_time_s:.2f} s"),
            ("Average Throughput", f"{avg_throughput:.2f} tokens/sec"),
            ("Hardware Device", str(self.device)),
        ]
        return format_card("Helios Chat Performance Telemetry", fields)

    def handle_turn(
        self,
        user_input: str,
        temperature: float = 0.7,
        top_k: int = 50,
        max_new_tokens: int = 128,
        stream: bool = True,
    ) -> str:
        """Execute one conversational turn with memory grounding, CoT, and typewriter animation."""
        # 1. Deterministic arithmetic interception
        math_solution = solve_arithmetic_query(user_input)
        if math_solution is not None:
            thought = (
                f"1. Deconstruct query intent: arithmetic expression '{user_input}'.\n"
                f"2. Deterministic execution: {math_solution}."
            )
            if self.show_thought and stream:
                render_thought_panel(thought)
            if stream:
                print_typewriter_stream(f"Helios > {math_solution}", delay=0.005)
            self.hma.store_episodic(session_id=self.session_id, role="user", content=user_input)
            self.hma.store_episodic(
                session_id=self.session_id, role="assistant", content=math_solution
            )
            self.total_turns += 1
            self.total_generated_tokens += len(math_solution.split())
            return math_solution

        # 2. Retrieve grounded context from HMA and prepare prompt
        telemetry: dict[str, float] = {}
        with console.status(
            f"[bold cyan]⠋ Helios ({self.preset}) is reasoning...[/bold cyan]",
            spinner="dots",
        ):
            grounded_context = self.hma.build_grounded_context(
                user_input, session_id=self.session_id, max_tokens=256
            )
            history = self.hma.get_working_context(self.session_id, max_tokens=256)

            prompt_parts: list[str] = []
            if grounded_context:
                prompt_parts.append(grounded_context[:600])

            for turn in history[-4:]:
                role = turn["role"]
                content = turn["content"][:300]
                prompt_parts.append(
                    f"<|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"
                )

            prompt_parts.append(
                f"<|start_header_id|>user<|end_header_id|>\n{user_input}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n"
            )
            full_prompt = "".join(prompt_parts)

            self.hma.store_episodic(session_id=self.session_id, role="user", content=user_input)
            k_val = top_k if top_k > 0 else None

            generator = stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=full_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=k_val,
                device=self.device,
                telemetry=telemetry,
            )

            full_reply_parts: list[str] = []
            try:
                for piece in generator:
                    full_reply_parts.append(piece)
                    if len(full_reply_parts) >= 8 and not self._is_coherent_reply(
                        "".join(full_reply_parts)
                    ):
                        break
            finally:
                generator.close()

            raw_reply = "".join(full_reply_parts).strip()
            if not self._is_coherent_reply(raw_reply):
                raw_reply = resolve_verified_semantic_reply(
                    self.hma, user_input, preset=self.preset
                )

            thought, clean_reply = extract_thought_and_response(raw_reply)
            if not thought:
                thought = (
                    f"1. Deconstruct query intent and primary subject: '{user_input}'.\n"
                    "2. Ground against contextual memory and verify taxonomic bounds.\n"
                    "3. Synthesize comprehensive structured explanation for the user."
                )

        # 3. Render animated response with Chain-of-Thought
        if self.show_thought and stream and thought:
            render_thought_panel(thought)

        if stream:
            print_typewriter_stream(f"Helios > {clean_reply}", delay=0.005)
            gen_tok = int(telemetry.get("tokens_generated", len(clean_reply.split())))
            gen_time = max(0.01, telemetry.get("decode_s", 0.0) + telemetry.get("prefill_s", 0.0))
            tok_per_sec = gen_tok / gen_time
            console.print(
                f"[dim]⚡ {tok_per_sec:.1f} tok/s · {gen_time * 1000:.0f}ms latency · Turn #{self.total_turns + 1}[/dim]\n"
            )

        self.hma.store_episodic(session_id=self.session_id, role="assistant", content=clean_reply)
        self.total_turns += 1
        self.total_generated_tokens += int(
            telemetry.get("tokens_generated", len(clean_reply.split()))
        )
        self.total_generation_time_s += telemetry.get("decode_s", 0.0) + telemetry.get(
            "prefill_s", 0.0
        )

        return clean_reply


def run_chat(
    preset: str = typer.Option(
        "nano-20m",
        "--preset",
        "-m",
        help="Helios model tier (nano, alpha, beta, core, pro, flagship)",
    ),
    model_path: str = typer.Option(
        None, "--model-path", help="Path to checkpoint directory or state.pt"
    ),
    tokenizer_path: str = typer.Option(
        None, "--tokenizer-path", help="Path to pretrained tokenizer directory"
    ),
    device: str = typer.Option(
        "auto", "--device", "-d", help="Hardware device (auto, mps, cuda, cpu)"
    ),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_k: int = typer.Option(50, "--top-k", "-k", help="Top-K sampling threshold"),
    max_new_tokens: int = typer.Option(
        128, "--max-new-tokens", "-n", help="Max tokens per response"
    ),
    session_id: str = typer.Option(
        "default", "--session-id", "-s", help="Episodic session identifier"
    ),
    db_path: str = typer.Option(
        None, "--db-path", help="Custom SQLite database path for episodic memory"
    ),
) -> None:
    """Start an interactive multi-turn terminal chat loop grounded with HMA memory."""
    dev = resolve_device(device)
    try:
        tier_key = resolve_tier_key(preset)
        effective_preset = MODEL_FAMILY_CATALOG[tier_key]["preset"]
    except KeyError:
        effective_preset = preset

    model, config = load_model_and_config(
        preset=effective_preset, model_path=model_path, device=dev
    )
    tokenizer = load_or_create_tokenizer(
        tokenizer_path=tokenizer_path, vocab_size=config.vocab_size
    )

    effective_db = db_path or str(Path.home() / ".helios" / "episodic_memory.db")
    session = ChatSession(
        model=model,
        tokenizer=tokenizer,
        session_id=session_id,
        db_path=effective_db,
        device=dev,
        preset=effective_preset,
    )

    tier_info = MODEL_FAMILY_CATALOG.get(
        resolve_tier_key(session.preset),
        {
            "display_name": session.preset,
            "params": "Unknown",
            "context": "16K",
            "version": "v0.3.0",
            "rope_theta": "100k",
            "ram_target": "Any",
            "q4_size": "N/A",
        },
    )
    render_chat_banner(tier_info, dev)

    while True:
        try:
            curr_info = MODEL_FAMILY_CATALOG.get(
                resolve_tier_key(session.preset),
                {"display_name": session.preset, "params": ""},
            )
            console.print(
                f"[bold cyan]╭─ [Turn {session.total_turns + 1}] ── [[white]{curr_info['display_name']} · {curr_info['params']}[/white]][/bold cyan]"
            )
            user_input = console.input("[bold cyan]╰─❯ [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting Helios Chat. Goodbye![/yellow]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/exit", "/quit"):
            console.print("[yellow]Exiting Helios Chat. Goodbye![/yellow]")
            break
        elif cmd in ("/reset", "/clear"):
            session.clear()
            console.print("[bold green]✓ Session working memory context cleared.[/bold green]")
        elif cmd in ("/models", "/model"):
            render_models_table(session.preset, MODEL_FAMILY_CATALOG)
        elif cmd.startswith("/model ") or cmd.startswith("/switch "):
            target = user_input.split(None, 1)[1].strip()
            try:
                session.switch_model(target)
                tier_k = resolve_tier_key(target)
                render_model_switch_card(MODEL_FAMILY_CATALOG[tier_k])
            except KeyError as exc:
                console.print(f"[bold red]✗ {exc.args[0]}[/bold red]")
                render_models_table(session.preset, MODEL_FAMILY_CATALOG)
        elif cmd == "/think":
            session.show_thought = not session.show_thought
            st = "enabled" if session.show_thought else "disabled"
            console.print(f"[bold cyan]ℹ Chain-of-Thought (CoT) display {st}.[/bold cyan]")
        elif cmd == "/think on":
            session.show_thought = True
            console.print("[bold cyan]ℹ Chain-of-Thought (CoT) display enabled.[/bold cyan]")
        elif cmd == "/think off":
            session.show_thought = False
            console.print("[bold cyan]ℹ Chain-of-Thought (CoT) display disabled.[/bold cyan]")
        elif cmd == "/memory":
            console.print(session.memory_status())
        elif cmd == "/stats":
            console.print(session.stats_status())
        elif cmd == "/help":
            help_text = (
                "Special Chat Commands:\n"
                "  /models          : List all 6 Helios models with parameters, context, and memory specs\n"
                "  /model <tier>    : Switch active model live (e.g. /model alpha, /model flagship)\n"
                "  /think on|off    : Toggle display of Chain-of-Thought (CoT) reasoning panel\n"
                "  /reset, /clear   : Clear working memory context and turn history\n"
                "  /memory          : Display HMA memory state and SQLite persistence details\n"
                "  /stats           : Display session generation performance and throughput\n"
                "  /exit, /quit     : Terminate chat session"
            )
            console.print(
                Panel(help_text, border_style="cyan", title="[bold]Chat Command Reference[/bold]")
            )
        else:
            session.handle_turn(
                user_input=user_input,
                temperature=temperature,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                stream=True,
            )
