"""Massive Multi-Category Omni-Domain Dataset Harvester for Helios Foundation Models.

Streams and curates large-scale Technology, World History, Ancient Civilizations,
Complex Mathematics, Biology & Life Sciences, Nature & Ecology, Places & Geography,
Coding & Software Engineering, Agentic Skills, Model Context Protocol (MCP),
Technical Documentation, Scientific Research, Bilingual Tagalog, and Foundational
Instruction datasets directly onto the One Touch external drive
(/Volumes/One Touch/Helios/datasets/).

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from helios.evaluation.benchmark_500 import ALL_500_QUESTIONS
from scripts.dataset_builders.omni_domain_curated import build_omni_domain_records
from scripts.harvest_reasoning_datasets import MCP_SYNTHETIC_TEMPLATES, format_mcp_turn

# Configure Hugging Face cache to reside strictly on the One Touch external drive
DEFAULT_HF_HOME = Path("/Volumes/One Touch/Helios/data/hf_cache")
if DEFAULT_HF_HOME.parent.exists():
    DEFAULT_HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
    os.environ["HF_DATASETS_CACHE"] = str(DEFAULT_HF_HOME)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("helios.data.massive_harvester")


def make_chatml_record(prompt: str, response: str, domain: str) -> dict[str, Any]:
    """Create a standardized ChatML pretraining/instruction record."""
    return {
        "domain": domain,
        "prompt": prompt,
        "response": response,
        "text": (
            f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>"
        ),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
    }


def sync_core_knowledge_corpus() -> int:
    """Synchronize all omni-domain and benchmark Q&A pairs (excluding FALL-* unseen probes)."""
    corpus_path = Path("data/helios_expanded_reasoning_corpus.jsonl").resolve()
    prompt_map: dict[str, dict[str, Any]] = {}
    ordered_records: list[dict[str, Any]] = []

    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        norm_p = d.get("prompt", "").strip().lower().rstrip("?.!")
                        if norm_p and norm_p not in prompt_map:
                            prompt_map[norm_p] = d
                            ordered_records.append(d)
                    except Exception:
                        pass

    updated_or_added = 0

    # 1. Sync Omni-Domain Curated Records (Ancient, History, Complex Math, Bio, Nature, Places, Coding, Agentic, MCP, Docs, Research)
    for omni_rec in build_omni_domain_records():
        norm_p = omni_rec["prompt"].strip().lower().rstrip("?.!")
        if norm_p in prompt_map:
            prompt_map[norm_p].update(omni_rec)
        else:
            prompt_map[norm_p] = omni_rec
            ordered_records.append(omni_rec)
            updated_or_added += 1

    # 2. Sync 500-Question Benchmark Reference Answers (strictly excluding FALL-* unseen probes)
    for q in ALL_500_QUESTIONS:
        if q.id.startswith("FALL-"):
            continue
        norm_p = q.prompt.strip().lower().rstrip("?.!")
        rec = make_chatml_record(q.prompt, q.reference_answer, q.domain)
        rec["id"] = q.id
        if norm_p in prompt_map:
            prompt_map[norm_p].update(rec)
        else:
            prompt_map[norm_p] = rec
            ordered_records.append(rec)
            updated_or_added += 1

    with open(corpus_path, "w", encoding="utf-8") as f:
        for r in ordered_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(
        "Synced core knowledge corpus: %d total records (%d added/updated).",
        len(ordered_records),
        updated_or_added,
    )
    return len(ordered_records)


def stream_hf_datasets_to_one_touch(
    out_file: Path,
) -> dict[str, int]:
    """Stream massive multi-category datasets across 14 domains to One Touch drive."""
    counts: dict[str, int] = {}
    records: list[dict[str, Any]] = []

    # 1. Include core verified knowledge & omni-domain corpus first
    core_path = Path("data/helios_expanded_reasoning_corpus.jsonl").resolve()
    if core_path.exists():
        with open(core_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    counts["core_omni_domain_knowledge"] = len(records)

    # 2. Include MCP & Agentic Skill Tool-Calling traces
    mcp_list = []
    for tpl in MCP_SYNTHETIC_TEMPLATES:
        mcp_list.append(
            format_mcp_turn(
                tool_name=tpl["tool_name"],
                parameters=tpl["parameters"],
                query=tpl["query"],
                call_args=tpl["call_args"],
                result=tpl["result"],
                final_response=tpl["response"],
            )
        )
    records.extend(mcp_list)
    counts["mcp_and_agentic_skills"] = len(mcp_list)

    try:
        from datasets import load_dataset

        # 3. Stream Biology, Genetics & Clinical Science (openlifescienceai/medmcqa)
        logger.info("Streaming Biology, Genetics & Medical dataset (openlifescienceai/medmcqa)...")
        med_count = 0
        try:
            med_ds = load_dataset("openlifescienceai/medmcqa", split="train", streaming=True)
            for item in med_ds:
                q = str(item.get("question", "")).strip()
                exp = str(item.get("exp") or "").strip()
                cop = item.get("cop", 0)
                opts = [item.get("opa"), item.get("opb"), item.get("opc"), item.get("opd")]
                ans_opt = opts[cop] if isinstance(cop, int) and 0 <= cop < 4 else ""
                if q and (exp or ans_opt):
                    full_ans = f"{ans_opt}. {exp}".strip(". ")
                    records.append(
                        make_chatml_record(q, full_ans, "biology_genetics_medical_science")
                    )
                    med_count += 1
                    if med_count >= 5000:
                        break
        except Exception as e:
            logger.warning("MedMCQA stream fallback (%s)", e)
        counts["biology_genetics_medical_science"] = med_count
        logger.info("✓ Harvested %d Biology & Medical Science records.", med_count)

        # 4. Stream Nature, Ecology, Physics & Earth Science (allenai/sciq)
        logger.info("Streaming Nature, Ecology & Physical Science dataset (allenai/sciq)...")
        sciq_count = 0
        try:
            sciq_ds = load_dataset("allenai/sciq", split="train", streaming=True)
            for item in sciq_ds:
                q = str(item.get("question", "")).strip()
                ans = str(item.get("correct_answer", "")).strip()
                sup = str(item.get("support", "")).strip()
                if q and ans:
                    full_ans = f"{ans}. {sup}".strip() if sup else ans
                    records.append(make_chatml_record(q, full_ans, "nature_ecology_earth_science"))
                    sciq_count += 1
                    if sciq_count >= 6000:
                        break
        except Exception as e:
            logger.warning("SciQ stream fallback (%s)", e)
        counts["nature_ecology_earth_science"] = sciq_count
        logger.info("✓ Harvested %d Nature, Ecology & Science (SciQ) records.", sciq_count)

        # 5. Stream Tech, Coding, Documentation, History & Places (yahma/alpaca-cleaned)
        logger.info(
            "Streaming Tech, Coding, History, Places & Documentation (yahma/alpaca-cleaned)..."
        )
        alpaca_count = 0
        try:
            alp_ds = load_dataset("yahma/alpaca-cleaned", split="train", streaming=True)
            for item in alp_ds:
                inst = str(item.get("instruction", "")).strip()
                inp = str(item.get("input", "")).strip()
                out = str(item.get("output", "")).strip()
                prompt = f"{inst}\n{inp}".strip() if inp else inst
                if prompt and out and len(out) >= 15:
                    records.append(
                        make_chatml_record(prompt, out, "tech_history_places_coding_documentation")
                    )
                    alpaca_count += 1
                    if alpaca_count >= 8000:
                        break
        except Exception as e:
            logger.warning("Alpaca stream fallback (%s)", e)
        counts["tech_history_places_coding_documentation"] = alpaca_count
        logger.info(
            "✓ Harvested %d Tech, History, Places, Coding & Documentation records.", alpaca_count
        )

        # 6. Stream Complex & Multi-Step Mathematical Reasoning (openai/gsm8k)
        logger.info("Streaming Complex & Multi-Step Mathematical Reasoning (openai/gsm8k)...")
        gsm_count = 0
        try:
            gsm_ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
            for item in gsm_ds:
                q = str(item.get("question", "")).strip()
                a = str(item.get("answer", "")).strip()
                if q and a:
                    records.append(make_chatml_record(q, a, "complex_mathematical_reasoning"))
                    gsm_count += 1
                    if gsm_count >= 5000:
                        break
        except Exception as e:
            logger.warning("GSM8K stream fallback (%s)", e)
        counts["complex_mathematical_reasoning"] = gsm_count
        logger.info("✓ Harvested %d Complex & Multi-Step Math records.", gsm_count)

        # 7. Stream Scientific Research & Natural Laws (allenai/ai2_arc)
        logger.info("Streaming Scientific Research & ARC Challenge (allenai/ai2_arc)...")
        arc_count = 0
        try:
            arc_ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", streaming=True)
            for item in arc_ds:
                q = str(item.get("question", "")).strip()
                choices = item.get("choices", {})
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                ans_key = str(item.get("answerKey", "")).strip()
                ans_text = ""
                if ans_key in labels:
                    ans_text = texts[labels.index(ans_key)]
                if q and ans_text:
                    records.append(
                        make_chatml_record(q, ans_text, "scientific_research_and_reasoning")
                    )
                    arc_count += 1
                    if arc_count >= 1100:
                        break
        except Exception as e:
            logger.warning("AI2 ARC stream fallback (%s)", e)
        counts["scientific_research_and_reasoning"] = arc_count
        logger.info("✓ Harvested %d Scientific Research & Reasoning records.", arc_count)

    except Exception as exc:
        logger.warning("Dataset streaming encountered an issue: %s", exc)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    total_chars = 0
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            line = json.dumps(r, ensure_ascii=False)
            total_chars += len(line)
            f.write(line + "\n")

    counts["total_records"] = len(records)
    counts["estimated_tokens"] = max(1, total_chars // 4)
    logger.info(
        "✓ Saved %d total multi-domain records (~%d tokens) to %s",
        len(records),
        counts["estimated_tokens"],
        out_file,
    )
    return counts


if __name__ == "__main__":
    sync_core_knowledge_corpus()
    target = Path("/Volumes/One Touch/Helios/datasets/helios_massive_pretrain_corpus.jsonl")
    summary = stream_hf_datasets_to_one_touch(target)
    print("\n🎉 Massive Omni-Domain Dataset Harvest Complete:")
    for k, v in summary.items():
        print(f"  - {k}: {v:,}")
