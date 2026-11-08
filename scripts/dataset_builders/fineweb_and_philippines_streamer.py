"""Streaming and Ingestion for FineWeb-Edu and Wikipedia Philippine Corpora.

Streams high-scoring educational web text from HuggingFaceFW/fineweb-edu,
encyclopedic Tagalog text from wikimedia/wikipedia (20231101.tl), and English
Wikipedia articles on Philippine history, institutions, and geopolitics.

Author: Arron Kian Parejas <parejasarronkian@gmail.com>
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("helios.data.fineweb_philippines")


def stream_fineweb_edu_records(limit: int = 5000) -> list[dict[str, Any]]:
    """Stream top educational web text samples from HuggingFaceFW/fineweb-edu."""
    logger.info("Streaming FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT)...")
    records: list[dict[str, Any]] = []
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
        )
        for item in ds:
            text = str(item.get("text", "")).strip()
            score = item.get("score", 0.0)
            # Prioritize high educational score text (score >= 2.5 or length >= 200 chars)
            if text and (score >= 2.5 or len(text) >= 200):
                title = text[:80].split("\n")[0].strip()
                prompt = f"Explain the core educational principles of: {title}"
                records.append(
                    {
                        "domain": "fineweb_educational_web_knowledge",
                        "prompt": prompt,
                        "response": text[:1500],
                        "text": (
                            f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
                            f"<|start_header_id|>assistant<|end_header_id|>\n{text[:1500]}<|eot_id|>"
                        ),
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": text[:1500]},
                        ],
                    }
                )
                if len(records) >= limit:
                    break
    except Exception as exc:
        logger.warning("FineWeb-Edu stream fallback (%s)", exc)
    logger.info("✓ Harvested %d FineWeb-Edu educational records.", len(records))
    return records


def stream_wikipedia_philippines_records(limit: int = 5000) -> list[dict[str, Any]]:
    """Stream Tagalog and Philippine-focused articles from wikimedia/wikipedia."""
    logger.info("Streaming Wikipedia Tagalog (wikimedia/wikipedia, 20231101.tl)...")
    records: list[dict[str, Any]] = []
    try:
        from datasets import load_dataset

        tl_ds = load_dataset("wikimedia/wikipedia", "20231101.tl", split="train", streaming=True)
        for item in tl_ds:
            title = str(item.get("title", "")).strip()
            body = str(item.get("text", "")).strip()
            if title and body and len(body) >= 100:
                prompt = f"Ano ang {title}?"
                ans = body[:1200]
                records.append(
                    {
                        "domain": "philippine_wikipedia_tagalog",
                        "prompt": prompt,
                        "response": ans,
                        "text": (
                            f"<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>"
                            f"<|start_header_id|>assistant<|end_header_id|>\n{ans}<|eot_id|>"
                        ),
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": ans},
                        ],
                    }
                )
                if len(records) >= limit:
                    break
    except Exception as exc:
        logger.warning("Wikipedia Tagalog stream fallback (%s)", exc)
    logger.info("✓ Harvested %d Wikipedia Tagalog records.", len(records))
    return records
