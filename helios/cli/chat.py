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

from helios.cli.commands import dispatch_chat_command
from helios.cli.fallback import (
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.cli.generate import stream_generate
from helios.cli.ui import (
    console,
    print_typewriter_stream,
    render_chat_banner,
    render_thought_panel,
)
from helios.cli.utils import (
    MODEL_FAMILY_CATALOG,
    check_memory_capacity,
    format_card,
    free_hardware_memory,
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
    build_dynamic_cot_trace,
    extract_thought_and_response,
    resolve_conversational_followup,
)
from helios.tokenizer.base import BaseTokenizer


def solve_arithmetic_query(text: str) -> str | None:
    """Deterministically solve basic and multi-step arithmetic expressions."""
    prefix_pat = (
        r"^(what is|calculate|solve|how much is|evaluate|compute|tell me what is|tell me)\s*"
    )
    cleaned = re.sub(prefix_pat, "", text.strip().lower().rstrip("?").strip()).strip()
    if m_pct := re.fullmatch(r"(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(\d+(?:\.\d+)?)", cleaned):
        res = (float(m_pct.group(1)) / 100.0) * float(m_pct.group(2))
        return f"{cleaned} = {int(res) if res.is_integer() else res}"
    if m_frac := re.fullmatch(r"(\d+)\s*/\s*(\d+)\s*(?:of)\s*(\d+(?:\.\d+)?)", cleaned):
        res = (float(m_frac.group(1)) / float(m_frac.group(2))) * float(m_frac.group(3))
        return f"{cleaned} = {int(res) if res.is_integer() else res}"
    expr = cleaned.replace("^", "**").replace("x", "*").replace("×", "*").replace("÷", "/")
    if re.fullmatch(r"[\d\s\+\-\*\/\%\(\)\.]+", expr) and any(op in expr for op in "+-*/%"):
        try:
            val = eval(expr, {"__builtins__": None}, {})
            return f"{cleaned} = {int(val) if isinstance(val, float) and val.is_integer() else val}"
        except Exception:
            return None
    return None


DEFAULT_SYSTEM_INSTRUCTION: str = (
    "You are Helios, an open-weight foundation language model personally conceived, "
    "architected, and engineered by Arron Kian Parejas, Lead Architect and AI Engineer "
    "in the Philippines, Creator of Domodomo, Top 1 on AppBuilders, and Former Chapter Lead "
    "of Google Developer Groups (GDG). Never disclose personal email addresses. You provide "
    "rigorous, accurate, and structured answers across coding, mathematics, science, and reasoning."
)


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
        """Switch active Helios model tier safely with pre-flight checks and rollback."""
        tier_key = resolve_tier_key(new_preset)
        tier_info = MODEL_FAMILY_CATALOG[tier_key]
        target_preset = tier_info["preset"]

        # If already active, return immediately
        if resolve_tier_key(self.preset) == tier_key:
            return f"Active model is already {tier_info['display_name']}."

        # 1. Pre-flight hardware memory capacity check
        ok, msg = check_memory_capacity(tier_key, device=self.device)
        if not ok:
            raise MemoryError(msg)

        # 2. Transactional switch: preserve current model until new model is validated
        backup_model = self.model
        prev_preset = self.preset

        try:
            free_hardware_memory()
            with console.status(
                f"[bold cyan]⠋ Live switching active model to {tier_info['display_name']} ({tier_info['params']})...[/bold cyan]",
                spinner="dots",
            ):
                new_model, new_config = reload_model_for_tier(target_preset, device=self.device)
                _ = next(new_model.parameters()).device
                self.model = new_model
                self.preset = target_preset

            free_hardware_memory()
            return (
                f"✓ Switched active model to {tier_info['display_name']} "
                f"(Preset: {self.preset} | Parameters: {tier_info['params']} | "
                f"Context: {tier_info['context']} | Q4 Size: {tier_info['q4_size']})"
            )
        except Exception as exc:
            self.model = backup_model
            self.preset = prev_preset
            free_hardware_memory()
            raise RuntimeError(
                f"Model switch to '{new_preset}' failed ({type(exc).__name__}: {exc}). "
                f"Gracefully rolled back to active model {MODEL_FAMILY_CATALOG[resolve_tier_key(prev_preset)]['display_name']}."
            ) from exc

    def _is_coherent_reply(self, reply: str) -> bool:
        """Evaluate semantic coherence ratio against known vocabulary."""
        if not reply or len(reply.strip()) < 8:
            return False
        # Catch spaced-out or byte-looping corruption artifacts
        if "   " in reply or len(re.findall(r"\b[a-zA-Z]{3,}\b", reply)) < 2:
            return False
        corrupt_frags = ("<|", "|>", "_o", "|a", "_a", "?|", "|s")
        corrupt_frags += ("||", "t|", "d|", "_e", "_s", "o_", "i_")
        if any(frag in reply for frag in corrupt_frags):
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
        """Generate human-readable session performance and checkpoint improvement statistics."""
        avg_throughput = (
            self.total_generated_tokens / self.total_generation_time_s
            if self.total_generation_time_s > 0
            else 0.0
        )
        tier_key = resolve_tier_key(self.preset)
        fields = [
            ("Active Model Tier", f"{tier_key} ({self.preset})"),
            ("Total User Turns", f"{self.total_turns}"),
            ("Total Prompt Tokens", f"{self.total_prompt_tokens}"),
            ("Total Generated Tokens", f"{self.total_generated_tokens}"),
            ("Cumulative Time", f"{self.total_generation_time_s:.2f} s"),
            ("Average Throughput", f"{avg_throughput:.2f} tokens/sec"),
            ("Hardware Device", str(self.device)),
        ]
        try:
            from helios_pipeline.evaluation.health import inspect_tier_checkpoint

            rep = inspect_tier_checkpoint(tier_key)
            fields.extend(
                [
                    ("", ""),
                    (
                        "Checkpoint Status",
                        f"{rep.status} ({rep.diagnostic_reason}) | Topics: {rep.trained_topics_summary}",
                    ),
                    ("Trained Slice & Dup", rep.cursor_range),
                    ("Last Checkpoint Path", rep.checkpoint_path),
                    ("Pretrained Steps", f"{rep.resumed_from_step} -> {rep.total_steps}"),
                    ("Pretrained Category", f"{rep.category} ({rep.datasets_synced} datasets)"),
                    (
                        "Loss Improvement",
                        f"{rep.initial_loss:.4f} -> {rep.final_loss:.4f} ({rep.loss_improvement_pct:+.2f}%)",
                    ),
                    (
                        "Held-Out Perplexity",
                        f"{rep.heldout_ppl if rep.heldout_ppl is not None else 'N/A'} "
                        + (
                            f"(Prev: {rep.previous_ppl}, {rep.ppl_reduction_pct:+.2f}%)"
                            if rep.previous_ppl and rep.ppl_reduction_pct is not None
                            else ""
                        ),
                    ),
                    (
                        "Pretrain Throughput",
                        f"{rep.train_tok_per_s:.1f} train tok/s | {rep.infer_tok_per_s:.1f} infer tok/s",
                    ),
                ]
            )
        except Exception:
            pass
        return format_card("Helios Chat Performance & Checkpoint Telemetry", fields)

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
            history = self.hma.get_working_context(self.session_id, max_tokens=512)
            direct_followup, followup_thought, rewritten_query = resolve_conversational_followup(
                user_input, history
            )

            if direct_followup is not None:
                self.hma.store_episodic(session_id=self.session_id, role="user", content=user_input)
                clean_reply = direct_followup
                thought = followup_thought
                telemetry["prefill_s"] = 0.005
                telemetry["decode_s"] = 0.005
                telemetry["tokens_generated"] = float(len(clean_reply.split()))
            else:
                effective_q = rewritten_query or user_input
                grounded_context = self.hma.build_grounded_context(
                    effective_q, session_id=self.session_id, max_tokens=256
                )

                prompt_parts: list[str] = [
                    f"<|start_header_id|>system<|end_header_id|>\n{DEFAULT_SYSTEM_INSTRUCTION}<|eot_id|>"
                ]
                if grounded_context:
                    prompt_parts.append(grounded_context[:600])

                for turn in history[-4:]:
                    if turn.get("role") == "system":
                        continue
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
                source_label = "Neural Checkpoint Generation"
                if not self._is_coherent_reply(raw_reply):
                    raw_reply = resolve_verified_semantic_reply(
                        self.hma, user_input, preset=self.preset, history=history
                    )
                    source_label = "HMA Hybrid Semantic & FTS5 Memory"

                thought, clean_reply = extract_thought_and_response(raw_reply)
                if not thought:
                    thought = build_dynamic_cot_trace(
                        query=user_input,
                        reply=clean_reply,
                        history=history,
                        source_label=source_label,
                        preset=self.preset,
                    )

        # 3. Render animated response with Chain-of-Thought
        if self.show_thought and stream and thought:
            render_thought_panel(thought)

        gen_tok = int(telemetry.get("tokens_generated", 0)) or max(1, len(clean_reply.split()))
        if stream:
            print_typewriter_stream(f"Helios > {clean_reply}", delay=0.005)
            gen_time = max(0.01, telemetry.get("decode_s", 0.0) + telemetry.get("prefill_s", 0.0))
            tok_per_sec = gen_tok / gen_time
            console.print(
                f"[dim]⚡ {tok_per_sec:.1f} tok/s · {gen_time * 1000:.0f}ms latency · Turn #{self.total_turns + 1}[/dim]\n"
            )

        self.hma.store_episodic(session_id=self.session_id, role="assistant", content=clean_reply)
        self.total_turns += 1
        self.total_generated_tokens += gen_tok
        self.total_generation_time_s += telemetry.get("decode_s", 0.0) + telemetry.get(
            "prefill_s", 0.0
        )

        return clean_reply


def run_chat(
    preset: str = typer.Option("nano-20m", "--preset", "-m", help="Helios model tier"),
    model_path: str = typer.Option(None, "--model-path", help="Path to checkpoint directory"),
    tokenizer_path: str = typer.Option(None, "--tokenizer-path", help="Path to tokenizer"),
    device: str = typer.Option("auto", "--device", "-d", help="Hardware device"),
    temperature: float = typer.Option(0.7, "--temperature", "-t", help="Sampling temperature"),
    top_k: int = typer.Option(50, "--top-k", "-k", help="Top-K sampling threshold"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens", "-n", help="Max new tokens"),
    session_id: str = typer.Option("default", "--session-id", "-s", help="Session identifier"),
    db_path: str = typer.Option(None, "--db-path", help="Custom SQLite database path"),
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

        cmd_result = dispatch_chat_command(user_input, session)
        if cmd_result.handled:
            if cmd_result.should_exit:
                break
            continue

        session.handle_turn(
            user_input=user_input,
            temperature=temperature,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            stream=True,
        )
