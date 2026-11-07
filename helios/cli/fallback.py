"""Subject-Aware Semantic Verification and Epistemic Fallback Engine for Helios Chat.

Prevents stopword-only BM25 false positives (such as returning '42' from 'What is 15 + 27?'
on 'What is Google?'), enforces prompt-subject token matching and English/Tagalog language
alignment, and returns an explicit epistemic fallback when a topic is outside the current
training session.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from helios.cli.utils import MODEL_FAMILY_CATALOG, resolve_tier_key
from helios.memory.bm25 import extract_content_tokens

if TYPE_CHECKING:
    from helios.memory.controller import HMAController

EPISTEMIC_FALLBACK_EN = (
    "This topic is not yet in my current training session. Please try asking another question!"
)

EPISTEMIC_FALLBACK_TL = (
    "Ang paksang ito ay wala pa sa aking kasalukuyang training session. "
    "Subukang magtanong ng ibang katanungan!"
)

TAGALOG_MARKERS: frozenset[str] = frozenset(
    {
        "ano",
        "sino",
        "saan",
        "kailan",
        "bakit",
        "paano",
        "alin",
        "ang",
        "mga",
        "sa",
        "ng",
        "na",
        "ay",
        "kamusta",
        "kumusta",
        "salamat",
        "magandang",
        "pilipinas",
        "gumawa",
        "lumikha",
        "sayo",
        "iyo",
        "ibig",
        "sabihin",
        "tawag",
        "salin",
        "kabisera",
        "pambansang",
    }
)


def is_tagalog_text(text: str) -> bool:
    """Detect whether a query or document is primarily written in Tagalog/Filipino."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return False
    # Check if first word is a Tagalog interrogative or greeting
    if tokens[0] in {
        "ano",
        "sino",
        "saan",
        "kailan",
        "bakit",
        "paano",
        "alin",
        "ilan",
        "anong",
        "kamusta",
        "kumusta",
        "salamat",
    }:
        return True
    tl_hits = sum(1 for t in tokens if t in TAGALOG_MARKERS)
    return tl_hits >= 2


def is_bare_numeric_response(response: str) -> bool:
    """Check if a response is just a bare number (e.g. '42') unsuitable for conceptual questions."""
    cleaned = response.strip().rstrip(".")
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned))


def seed_hma_knowledge_base(hma: HMAController, valid_words: set[str]) -> int:
    """Seed HMA semantic index from local and One Touch external reasoning corpora."""
    candidates = [
        Path("data/helios_expanded_reasoning_corpus.jsonl").resolve(),
        Path("/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl").resolve(),
        Path("/Volumes/One Touch/Helios/data/helios_expanded_reasoning_corpus.jsonl").resolve(),
        Path("data/helios_contextual_reasoning.jsonl").resolve(),
        Path(__file__).parent.parent.parent / "data" / "helios_expanded_reasoning_corpus.jsonl",
        Path(__file__).parent.parent.parent / "data" / "helios_contextual_reasoning.jsonl",
    ]
    indexed_prompts: set[str] = set()
    doc_count = 0

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    p = data.get("prompt", "").strip()
                    r = data.get("response", "").strip()
                    if not p or not r:
                        continue
                    norm_p = p.lower().rstrip("?.!")
                    if norm_p in indexed_prompts:
                        continue
                    # Skip bare numeric responses for Q&A indexing since solve_arithmetic_query handles math
                    if is_bare_numeric_response(r):
                        continue
                    indexed_prompts.add(norm_p)
                    doc_count += 1
                    # Index prompt tokens heavily by repeating prompt in indexed text
                    hma.index_semantic(
                        doc_id=f"kb_doc_{doc_count}",
                        text=f"Q: {p} {p}\nA: {r}",
                        metadata={
                            "prompt": p,
                            "response": r,
                            "is_tagalog": is_tagalog_text(p),
                            "prompt_tokens": extract_content_tokens(p),
                        },
                    )
                    for w in (p + " " + r).split():
                        clean = w.lower().strip(".,!?:;\"'()[]{}")
                        if clean:
                            valid_words.add(clean)
                    # Cap in-memory BM25 seeding at 8,000 high-priority QA records for <45ms interactive latency
                    if doc_count >= 8000:
                        return doc_count
        except Exception:
            pass
    return doc_count


def resolve_identity_or_greeting(query: str, preset: str = "nano-20m") -> str | None:
    """Handle deterministic conversational greetings, model version, and creator queries."""
    q_lower = query.lower().strip().rstrip("?.!")
    try:
        tier_key = resolve_tier_key(preset)
        tier_info = MODEL_FAMILY_CATALOG[tier_key]
        display_name = tier_info["display_name"]
        params = tier_info["params"]
        context = tier_info["context"]
    except KeyError:
        display_name = f"Helios ({preset})"
        params = "17.84M"
        context = "2,048 tokens"

    # 1. Origin of the name Helios / About Helios
    origin_patterns = [
        "what is helios",
        "why is it named helios",
        "origin of helios",
        "meaning of helios",
        "sun god",
        "ano ang helios",
    ]
    if any(pat in q_lower for pat in origin_patterns):
        return (
            f"Helios is named after Helios (Ἥλιος), the ancient Greek Titan god and personification "
            f"of the Sun who drove his radiant chariot across the sky to bring light, clarity, and "
            f"truth to the world. Personally engineered and developed from the ground up by AI Engineer "
            f"Arron Kian Parejas (parejasarronkian@gmail.com), the Helios open-weight LLM family "
            f"(currently active: {display_name}, {params} parameters, {context} context) brings "
            f"transparent, hardware-efficient intelligence with Grouped-Query Attention (GQA), "
            f"RoPE scaling, SwiGLU, and Hierarchical Memory Architecture (HMA)."
        )

    # 2. Model version queries
    version_patterns = [
        "model version",
        "what version",
        "which version",
        "anong bersyon",
        "version of helios",
    ]
    if any(pat in q_lower for pat in version_patterns):
        return (
            f"I am running {display_name} v0.2.0 (Preset: {preset}, Parameters: {params}, "
            f"Context: {context}) with Hierarchical Memory Architecture (HMA) grounding, "
            f"personally engineered by AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com)."
        )

    # 3. Model identity queries
    identity_patterns = [
        "what model are you",
        "who are you",
        "what are you",
        "introduce yourself",
        "tell me about yourself",
        "anong model ka",
        "sino ka",
    ]
    if any(pat in q_lower for pat in identity_patterns):
        if is_tagalog_text(q_lower):
            return (
                f"Ako si {display_name} (Preset: {preset}, Parameters: {params}, Bersyon: v0.2.0), "
                "isang open-weight foundation language model na personal na dinisenyo at binuo ni "
                "AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com)."
            )
        return (
            f"I am {display_name} (Preset: {preset}, Parameters: {params}, Context: {context}, "
            "Version: v0.2.0), an open-weight foundation language model personally engineered and "
            "developed by AI Engineer Arron Kian Parejas (parejasarronkian@gmail.com)."
        )

    # 4. Creator / Developer queries
    creator_terms = [
        "who created you",
        "who made you",
        "who built you",
        "who is your creator",
        "who is your developer",
        "who developed you",
        "who designed you",
        "sino gumawa sayo",
        "sino ang gumawa sayo",
        "sino lumikha sayo",
        "sino ang lumikha sayo",
    ]
    if any(term in q_lower for term in creator_terms):
        if is_tagalog_text(q_lower):
            return (
                "Ako ay personal na nilikha at binuo ni AI Engineer Arron Kian Parejas "
                "(parejasarronkian@gmail.com)."
            )
        return (
            "I was personally designed and developed by AI Engineer Arron Kian Parejas "
            "(parejasarronkian@gmail.com), who engineered my transformer architecture, tokenizer, "
            "training pipeline, and Hierarchical Memory Architecture (HMA)."
        )

    # 4. Gratitude / Salamat
    if re.search(r"\b(salamat|maraming salamat)\b", q_lower):
        return "Walang anuman! Ikinagagalak kitang tulungan. Sabihin mo lamang kung may iba ka pang kailangan."
    if (
        re.search(r"\b(thank you|thanks)\b", q_lower)
        and "translate" not in q_lower
        and "salin" not in q_lower
    ):
        return "You're welcome! Feel free to ask another question anytime."

    # 5. Greetings
    if re.fullmatch(
        r"(hello|hi|hey|greetings|good morning|good afternoon|good evening|kamusta|kumusta)(\s+there|\s+helios|\s+po)?",
        q_lower,
    ):
        if is_tagalog_text(q_lower):
            return "Kamusta! Ako si Helios, handang tumulong sa iyong mga katanungan. Ano ang maitutulong ko sa iyo ngayon?"
        return "Hello! I am operating smoothly and ready to assist you. How can I help you today?"

    return None


def resolve_verified_semantic_reply(
    hma: HMAController,
    user_input: str,
    preset: str = "nano-20m",
) -> str:
    """Retrieve a verified semantic response or return the honest epistemic fallback."""
    # 1. Check identity, version, or greeting first
    special = resolve_identity_or_greeting(user_input, preset=preset)
    if special is not None:
        return special

    norm_q = user_input.strip().lower().rstrip("?.!")
    query_is_tl = is_tagalog_text(user_input)

    # 2. Exact normalized prompt match
    for doc in hma.semantic.documents.values():
        meta_p = doc.metadata.get("prompt", "").strip().lower().rstrip("?.!")
        if meta_p == norm_q:
            resp = doc.metadata.get("response", "")
            if resp and not is_bare_numeric_response(resp):
                return resp

    # 3. Subject-token verified prompt matching
    q_tokens = extract_content_tokens(user_input)
    if not q_tokens:
        return EPISTEMIC_FALLBACK_TL if query_is_tl else EPISTEMIC_FALLBACK_EN

    q_set = set(q_tokens)
    best_score = 0.0
    best_response: str | None = None

    for doc in hma.semantic.documents.values():
        p_tokens = doc.metadata.get("prompt_tokens")
        if p_tokens is None:
            p_tokens = extract_content_tokens(doc.metadata.get("prompt", ""))
        if not p_tokens:
            continue

        p_set = set(p_tokens)
        common = q_set & p_set
        if not common:
            continue

        # Fraction of user's content tokens covered by candidate prompt
        query_coverage = len(common) / len(q_set)
        # Fraction of candidate prompt's content tokens covered by user query (specificity)
        prompt_specificity = len(common) / len(p_set)

        # Require strong subject coverage:
        # For 1-2 content token queries (e.g. 'what is google?', 'who is newton?'), require 100% query coverage
        # and high prompt specificity so 'what is earth?' matches 'What is Earth?' instead of 'gravity on Earth'
        if len(q_set) <= 2:
            if query_coverage < 1.0 or prompt_specificity < 0.50:
                continue
        else:
            if query_coverage < 0.75:
                continue

        doc_is_tl = bool(doc.metadata.get("is_tagalog", False))
        lang_bonus = 0.25 if (doc_is_tl == query_is_tl) else -0.35

        score = (query_coverage * 0.65) + (prompt_specificity * 0.35) + lang_bonus
        if score > best_score:
            resp = doc.metadata.get("response", "")
            if resp and not is_bare_numeric_response(resp):
                best_score = score
                best_response = resp

    if best_response is not None and best_score >= 0.70:
        return best_response

    # 4. Honest Epistemic Fallback (never return '42' or unrelated guesses)
    return EPISTEMIC_FALLBACK_TL if query_is_tl else EPISTEMIC_FALLBACK_EN
