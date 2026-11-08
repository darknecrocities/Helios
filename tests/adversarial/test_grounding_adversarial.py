"""Tier-5 Adversarial Stress Test Suite for Milestone 8.

Empirically stresses:
1. Synthetic probe battery: verifies that all 15 FALL probes (FALL-01 to FALL-15),
   blorgonx99, zyxwvut123, xenofluvir-904, and zorglubin-beta reliably trigger exact
   epistemic refusal strings (EPISTEMIC_FALLBACK_EN / EPISTEMIC_FALLBACK_TL) with 0 false positives.
2. Real-world factual battery: tests 29 diverse factual queries across 8 foundational domains
   verifying genuine factual grounding and zero false epistemic refusals from records beyond 8,000.
3. Zero-leakage verification: ensures conversational replies do not leak mock tool schemas,
   synthetic JSON objects, control tokens, or <functioncall> blocks.
4. Streaming DataLoader stress test: stresses StreamingTextDataLoader across multiple interruption
   and resumption points and epoch crossings, asserting bit-accurate deterministic token batches.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pytest
import torch

from helios.cli.fallback import (
    EPISTEMIC_FALLBACK_EN,
    EPISTEMIC_FALLBACK_TL,
    resolve_verified_semantic_reply,
    seed_hma_knowledge_base,
)
from helios.memory.controller import HMAController
from helios.memory.episodic import EpisodicMemory
from helios.memory.fts_corpus import FTSCorpus, resolve_corpus_db_path
from helios.reasoning.cot_engine import is_synthetic_benchmark_probe
from helios.training.dataloader import StreamingTextDataLoader


@pytest.fixture(scope="module")
def grounded_hma() -> HMAController:
    """Initialize HMA controller grounded with full SQLite FTS5 corpus."""
    hma = HMAController(episodic_memory=EpisodicMemory(db_path=":memory:"))
    seed_hma_knowledge_base(hma, set())
    return hma


# ==============================================================================
# 1. Comprehensive Synthetic Probe Battery & Epistemic Refusal Tests
# ==============================================================================


@pytest.mark.tier5
def test_all_15_fall_probes_trigger_exact_epistemic_refusal(
    grounded_hma: HMAController,
) -> None:
    """Verify all 15 synthetic probes (FALL-01 to FALL-15) trigger exact refusal strings."""
    # fmt: off
    fall_probes: list[tuple[str, str, str]] = [
        ("FALL-01", "What is the capital city of the fictional exoplanet Xylophia-99?", EPISTEMIC_FALLBACK_EN),
        ("FALL-02", "What is the molecular weight of synthetic Chronotonium-842?", EPISTEMIC_FALLBACK_EN),
        ("FALL-03", "Who won the 2194 Galactic Hyperloop Championship on Titan?", EPISTEMIC_FALLBACK_EN),
        ("FALL-04", "What is the syntax of the uninvented BlorgonScript-77 compiler?", EPISTEMIC_FALLBACK_EN),
        ("FALL-05", "What is the GDP of the mythical underwater kingdom of Zephaniah-Prime?", EPISTEMIC_FALLBACK_EN),
        ("FALL-06", "Who is Professor Quasimoto Vexley of the Nebula Institute?", EPISTEMIC_FALLBACK_EN),
        ("FALL-07", "What is the boiling point of liquid Krypthonite-Zeta?", EPISTEMIC_FALLBACK_EN),
        ("FALL-08", "Where is the secret vault of Commander Zorblax located?", EPISTEMIC_FALLBACK_EN),
        ("FALL-09", "What is the default port of the HyperQuantum-TP99 protocol?", EPISTEMIC_FALLBACK_EN),
        ("FALL-10", "What is the half-life of Subspace Isotope Vespene-404?", EPISTEMIC_FALLBACK_EN),
        ("FALL-11", "Ano ang kabisera ng kathang-isip na planetang Zorgon-77?", EPISTEMIC_FALLBACK_TL),
        ("FALL-12", "Sino ang nag-imbento ng makinaryang HyperDrive-X999?", EPISTEMIC_FALLBACK_TL),
        ("FALL-13", "Ano ang sangkap ng mahiwagang gayuma ng Eldoria-55?", EPISTEMIC_FALLBACK_TL),
        ("FALL-14", "Saan matatagpuan ang nawawalang lungsod ng Atlantika-Omega?", EPISTEMIC_FALLBACK_TL),
        ("FALL-15", "Ano ang kahulugan ng salitang alien na Xyphlorian-88?", EPISTEMIC_FALLBACK_TL),
    ]
    # fmt: on

    for probe_id, prompt, expected_refusal in fall_probes:
        assert is_synthetic_benchmark_probe(prompt), f"Probe {probe_id} not recognized by detector"
        reply = resolve_verified_semantic_reply(grounded_hma, prompt, preset="nano-20m")
        assert reply == expected_refusal, (
            f"Probe {probe_id} failed exact refusal match. Expected: {expected_refusal!r}, Got: {reply!r}"
        )


@pytest.mark.tier5
def test_held_out_synthetic_probes_epistemic_refusal(
    grounded_hma: HMAController,
) -> None:
    """Verify blorgonx99, zyxwvut123, xenofluvir-904, and zorglubin-beta trigger refusal."""
    # fmt: off
    held_out_probes: list[tuple[str, str, str]] = [
        ("blorgonx99-en", "What is blorgonx99 and how is it used?", EPISTEMIC_FALLBACK_EN),
        ("blorgonx99-tl", "Ano ang blorgonx99 sa agham?", EPISTEMIC_FALLBACK_TL),
        ("zyxwvut123-en", "Explain the history of zyxwvut123.", EPISTEMIC_FALLBACK_EN),
        ("zyxwvut123-tl", "Ano ang ibig sabihin ng zyxwvut123?", EPISTEMIC_FALLBACK_TL),
        ("xenofluvir-en", "What are the clinical indications of xenofluvir-904?", EPISTEMIC_FALLBACK_EN),
        ("xenofluvir-tl", "Ano ang gamot na xenofluvir-904?", EPISTEMIC_FALLBACK_TL),
        ("zorglubin-en", "Is zorglubin-beta approved by the FDA?", EPISTEMIC_FALLBACK_EN),
        ("zorglubin-tl", "Saan ginagamit ang zorglubin-beta?", EPISTEMIC_FALLBACK_TL),
    ]
    # fmt: on

    for probe_id, prompt, expected_refusal in held_out_probes:
        assert is_synthetic_benchmark_probe(prompt), (
            f"{probe_id} must be recognized as synthetic probe"
        )
        reply = resolve_verified_semantic_reply(grounded_hma, prompt, preset="nano-20m")
        assert reply == expected_refusal, f"{probe_id} failed refusal match: got {reply!r}"


@pytest.mark.tier5
def test_synthetic_probe_zero_false_positives() -> None:
    """Assert genuine real-world queries never falsely trigger synthetic probe markers."""
    real_queries = [
        "What is a xylophone musical instrument?",
        "How does a hyperloop train work?",
        "Tell me about Titan the moon of Saturn.",
        "What is krypton gas on the periodic table?",
        "What is a chronometer clock?",
        "What is quantum computing?",
        "Who is commander in chief of the military?",
        "What is an isotope in nuclear physics?",
        "What is hard drive in computer storage?",
        "What is hyperdrive theory in science fiction?",
    ]
    for q in real_queries:
        assert not is_synthetic_benchmark_probe(q), (
            f"False positive probe detection for real query: {q}"
        )


@pytest.mark.tier5
def test_adversarial_zephaniah_false_positive_probe() -> None:
    """Empirical probe capturing false-positive refusal on Book of Zephaniah."""
    query = "What is the book of Zephaniah in the Bible?"
    # The detector should NOT treat the biblical book of Zephaniah as a synthetic benchmark probe
    assert not is_synthetic_benchmark_probe(query), (
        "False positive: 'zephaniah' marker caught legitimate query about biblical Book of Zephaniah"
    )


# ==============================================================================
# 2. Real-World Factual Battery Across 8 Domains (Beyond Record 8,000)
# ==============================================================================


@pytest.mark.tier5
def test_real_world_factual_battery_across_domains(
    grounded_hma: HMAController,
) -> None:
    """Test at least 25 real-world queries asserting genuine answers and ZERO false refusals."""
    # fmt: off
    factual_battery: list[tuple[str, str, str]] = [
        # History
        ("hist_1", "How did Julius Caesar die?", "44 bc"),
        ("hist_2", "Who was the first Emperor of Rome?", "augustus"),
        ("hist_3", "When did the storming of the Bastille take place during the French Revolution?", "1789"),
        ("hist_4", "Who wrote the 95 Theses sparking the Protestant Reformation?", "martin luther"),
        # Anatomy
        ("anat_1", "What is the largest organ in the human body?", "skin"),
        ("anat_2", "How many chambers does the human heart have?", "four"),
        ("anat_3", "What is the longest and strongest bone in the human skeleton?", "femur"),
        ("anat_4", "Which organ produces insulin in the human body?", "pancreas"),
        # Astrophysics
        ("astr_1", "What is the theoretical boundary around a black hole beyond which nothing can escape?", "event horizon"),
        ("astr_2", "What is the speed of light in a vacuum in meters per second?", "299"),
        ("astr_3", "What is the Chandrasekhar limit for the maximum mass of a stable white dwarf star?", "1.4"),
        ("astr_4", "What is the primary nuclear fusion reaction powering the Sun?", "fusion"),
        # Geometry
        ("geom_1", "What is the sum of the interior angles of a triangle in Euclidean geometry?", "180"),
        ("geom_2", "What theorem states that in a right triangle a squared plus b squared equals c squared?", "pythagor"),
        ("geom_3", "What is the formula for the area of a circle with radius r?", "pi"),
        # Organic Chemistry
        ("chem_1", "What functional group characterizes an alcohol molecule?", "hydroxyl"),
        ("chem_2", "What is the hybridization of carbon atoms in a benzene ring?", "sp2"),
        ("chem_3", "What rule states that in electrophilic addition to an alkene the hydrogen adds to the carbon with more hydrogens?", "markovnikov"),
        # Philippine Geography
        ("phil_1", "What is the capital city of the Philippines?", "manila"),
        ("phil_2", "What is the highest mountain peak in the Philippines?", "apo"),
        ("phil_3", "In which province are the famous Banaue Rice Terraces located?", "ifugao"),
        ("phil_4", "What is the famous active volcano with a near-perfect cone shape in Albay Philippines?", "mayon"),
        # Computer Science
        ("cs_1", "What is the worst-case time complexity of binary search on a sorted array?", "log"),
        ("cs_2", "Which theoretical model of computation uses a tape head and state transition table?", "turing"),
        ("cs_3", "What protocol uses SYN, SYN-ACK, and ACK to establish a reliable connection?", "tcp"),
        ("cs_4", "Which algorithm finds the shortest path between nodes in a graph with non-negative edge weights?", "dijkstra"),
        # Medicine
        ("med_1", "Which beta1 selective blocker is used in glaucoma?", "betaxolol"),
        ("med_2", "What is the normal physiological pH range of human arterial blood?", "7.35"),
        ("med_3", "Which class of antibiotics was discovered by Alexander Fleming from a Penicillium mold?", "penicillin"),
    ]
    # fmt: on

    assert len(factual_battery) >= 25, f"Must have >=25 queries, found {len(factual_battery)}"

    false_refusals: list[str] = []
    keyword_matches: int = 0

    for query_id, query, expected_kw in factual_battery:
        reply = resolve_verified_semantic_reply(grounded_hma, query, preset="nano-20m")
        reply_lower = reply.lower()

        # Epistemic refusal check (ZERO false refusals permitted)
        if (
            EPISTEMIC_FALLBACK_EN.lower() in reply_lower
            or EPISTEMIC_FALLBACK_TL.lower() in reply_lower
            or "not yet in my current training session" in reply_lower
            or "wala pa sa aking kasalukuyang training session" in reply_lower
        ):
            false_refusals.append(f"{query_id}: {query}")

        if expected_kw.lower() in reply_lower:
            keyword_matches += 1

    assert len(false_refusals) == 0, (
        f"Detected false epistemic refusals on real queries: {false_refusals}"
    )
    assert keyword_matches >= 18, (
        f"Expected at least 18 exact factual keyword grounding matches, got {keyword_matches}/{len(factual_battery)}"
    )


@pytest.mark.tier5
def test_fts5_reach_beyond_record_8000_and_sub_10ms_latency() -> None:
    """Verify FTS5 corpus indexes all 134,728 records and responds in sub-10ms latency."""
    db_path = resolve_corpus_db_path()
    corpus = FTSCorpus(db_path=db_path)
    total_records = corpus.count_records()
    assert total_records >= 134_000, f"Expected 134k+ records, got {total_records}"

    # Pre-warm connection to account for external USB disk spin-up
    for _ in range(3):
        _ = corpus.query_corpus("warmup query", top_k=1)

    deep_probe_queries = [
        "Julius Caesar 44 BC",
        "Betaxolol glaucoma ocular",
        "photosynthesis plants sunlight",
        "95 Theses Martin Luther",
        "black hole event horizon",
        "benzene hybridization sp2",
    ]

    high_index_hits = 0
    latencies: list[float] = []

    for query in deep_probe_queries:
        t0 = time.perf_counter()
        hits = corpus.query_corpus(query, top_k=3)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        assert len(hits) >= 1, f"Expected at least 1 hit for '{query}'"

        for h in hits:
            # Check for doc IDs beyond record 8,000
            m = re.search(r"doc_(\d+)", h.id)
            if m and int(m.group(1)) > 8000:
                high_index_hits += 1

    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 10.0, f"Average query latency exceeded 10ms SLA: {avg_latency:.2f}ms"
    assert high_index_hits >= 4, (
        f"Expected at least 4 deep corpus hits beyond record 8,000, got {high_index_hits}"
    )
    corpus.close()


# ==============================================================================
# 3. Zero-Leakage Verification
# ==============================================================================


@pytest.mark.tier5
def test_zero_leakage_factual_and_conversational_replies(
    grounded_hma: HMAController,
) -> None:
    """Verify conversational responses do not leak control tokens, schemas, or raw JSON."""
    test_queries = [
        "Hello! How are you doing today?",
        "Who created you?",
        "What is the capital of the Philippines?",
        "How did Julius Caesar die?",
        "What is photosynthesis?",
        "Which beta1 selective blocker is used in glaucoma?",
        "What is the Pythagorean theorem?",
        "Tell me about Albert Einstein.",
        "What is the worst-case time complexity of binary search?",
        "Why is the sky blue?",
    ]

    forbidden_patterns = [
        re.compile(r"<function_?call>", re.IGNORECASE),
        re.compile(r"<\|[a-zA-Z0-9_:\-]+\|>"),
        re.compile(r"\{\s*\"name\"\s*:\s*\"[^\"]+\"\s*,\s*\"parameters\"", re.IGNORECASE),
        re.compile(r"\{\s*\"type\"\s*:\s*\"function\"", re.IGNORECASE),
    ]

    for q in test_queries:
        reply = resolve_verified_semantic_reply(grounded_hma, q, preset="nano-20m")
        for pat in forbidden_patterns:
            assert not pat.search(reply), (
                f"Leakage detected in reply for '{q}':\nFound pattern {pat.pattern} in {reply!r}"
            )
        # Ensure reply is not raw unparsed JSON object
        trimmed = reply.strip()
        assert not (trimmed.startswith("{") and trimmed.endswith("}")), (
            f"Raw unparsed JSON object returned for '{q}': {reply}"
        )


@pytest.mark.tier5
def test_adversarial_agentic_query_leakage_probe(
    grounded_hma: HMAController,
) -> None:
    """Adversarially probe whether agentic tool-use dataset entries leak into replies."""
    agentic_tool_queries = [
        "What is the stock price of Apple?",
        "Call function check_palindrome",
        "Execute function get_stock_price",
    ]
    for q in agentic_tool_queries:
        reply = resolve_verified_semantic_reply(grounded_hma, q, preset="nano-20m")
        # If raw functioncall leaks, this assertion fails, capturing the empirical defect
        assert "<functioncall>" not in reply, (
            f"Raw <functioncall> leaked in reply for '{q}': {reply[:80]}"
        )
        assert "<|endoftext|>" not in reply, (
            f"Control token <|endoftext|> leaked in reply for '{q}'"
        )


# ==============================================================================
# 4. Streaming DataLoader Stress Test (Deterministic Resumption & Invariants)
# ==============================================================================


@pytest.mark.tier5
def test_streaming_dataloader_multi_point_resumption_stress(
    tmp_path: Path,
) -> None:
    """Stress test StreamingTextDataLoader with multiple interruptions and epoch crossing."""
    dataset_file = tmp_path / "stress_corpus.jsonl"
    num_records = 80
    with open(dataset_file, "w", encoding="utf-8") as f:
        for i in range(num_records):
            record = {
                "id": f"rec_{i:04d}",
                "domain": "adversarial_stress",
                "prompt": f"Adversarial query #{i} testing deterministic streaming guarantees.",
                "response": f"Adversarial factual reply #{i} verifying bit-accurate continuation.",
            }
            f.write(json.dumps(record) + "\n")

    batch_size = 4
    seq_len = 32
    max_epochs = 3

    # Primary uninterrupted reference stream
    primary_loader = StreamingTextDataLoader(
        dataset_file,
        batch_size=batch_size,
        seq_len=seq_len,
        max_epochs=max_epochs,
    )

    # Segment 1: Advance 3 batches and checkpoint
    _ = [next(primary_loader) for _ in range(3)]
    state_ckpt_1 = primary_loader.get_state()
    assert state_ckpt_1["batch_idx"] == 3
    assert state_ckpt_1["global_step"] == 3
    assert state_ckpt_1["token_offset"] == 3 * (batch_size * seq_len)

    # Segment 2: Advance next 5 batches on primary and checkpoint
    seg2_primary = [next(primary_loader) for _ in range(5)]
    state_ckpt_2 = primary_loader.get_state()
    assert state_ckpt_2["batch_idx"] == 8

    # Resume secondary loader from checkpoint 1 -> verify exact match with Segment 2
    resumed_loader_1 = StreamingTextDataLoader(
        dataset_file,
        batch_size=batch_size,
        seq_len=seq_len,
        max_epochs=max_epochs,
    )
    resumed_loader_1.load_state(state_ckpt_1)
    seg2_resumed = [next(resumed_loader_1) for _ in range(5)]

    for idx, (b_orig, b_res) in enumerate(zip(seg2_primary, seg2_resumed, strict=True)):
        assert torch.equal(b_orig["input_ids"], b_res["input_ids"]), (
            f"Bitwise tensor mismatch at step {idx} during Resumption 1"
        )
        assert torch.equal(b_orig["labels"], b_res["labels"])
    assert resumed_loader_1.cursor.batch_idx == 8
    assert resumed_loader_1.cursor.token_offset == primary_loader.cursor.token_offset

    # Segment 3: Advance 7 batches across epoch boundary
    seg3_primary = [next(primary_loader) for _ in range(7)]

    # Resume tertiary loader from checkpoint 2 -> verify exact match with Segment 3
    resumed_loader_2 = StreamingTextDataLoader(
        dataset_file,
        batch_size=batch_size,
        seq_len=seq_len,
        max_epochs=max_epochs,
    )
    resumed_loader_2.load_state(state_ckpt_2)
    seg3_resumed = [next(resumed_loader_2) for _ in range(7)]

    for idx, (b_orig, b_res) in enumerate(zip(seg3_primary, seg3_resumed, strict=True)):
        assert torch.equal(b_orig["input_ids"], b_res["input_ids"]), (
            f"Bitwise tensor mismatch at step {idx} during Resumption 2"
        )
        assert torch.equal(b_orig["labels"], b_res["labels"])

    assert resumed_loader_2.cursor.epoch == primary_loader.cursor.epoch
    assert resumed_loader_2.cursor.batch_idx == primary_loader.cursor.batch_idx
    assert resumed_loader_2.cursor.total_tokens_seen == primary_loader.cursor.total_tokens_seen


@pytest.mark.tier5
def test_streaming_dataloader_seek_and_reset(tmp_path: Path) -> None:
    """Verify seek and reset operations restore deterministic coordinates."""
    dataset_file = tmp_path / "seek_corpus.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(json.dumps({"prompt": f"Q{i}", "response": f"A{i}"}) + "\n")

    loader = StreamingTextDataLoader(dataset_file, batch_size=2, seq_len=16, max_epochs=2)
    _ = [next(loader) for _ in range(3)]
    assert loader.cursor.batch_idx == 3

    loader.reset()
    assert loader.cursor.batch_idx == 0
    assert loader.cursor.epoch == 0
    assert loader.cursor.sample_cursor == 0
    assert len(loader._token_buffer) == 0

    loader.seek(sample_cursor=5, token_offset=160, epoch=1)
    assert loader.cursor.sample_cursor == 5
    assert loader.cursor.token_offset == 160
    assert loader.cursor.epoch == 1
    assert len(loader._token_buffer) == 0
