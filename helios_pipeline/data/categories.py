"""Category-aware dataset registry and corpus filter for HELIOS continuous pretraining.

Maps user-selectable categories (`all`, `fineweb`, `math`, `history`, `coding`,
`science`, `geography`, `law`, `agents`, `tagalog`) to the 177 external datasets in
`sot/datasets_sot.json` and the 120,616-record OneTouch pretraining corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CategorySpec:
    """Metadata and matching rules for a pretraining dataset category."""

    key: str
    display_name: str
    description: str
    sot_prefixes: tuple[str, ...]
    corpus_breakdown_keys: tuple[str, ...]
    keywords: tuple[str, ...]
    cot_samples: tuple[str, ...]


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "all": CategorySpec(
        key="all",
        display_name="All Omni-Domain (177 Datasets)",
        description="Complete 16-domain pretraining mix across FineWeb, Math, History, Code, Science, MCP & Tagalog",
        sot_prefixes=tuple(f"{i}." for i in range(1, 17)),
        corpus_breakdown_keys=(
            "core_omni_domain_knowledge",
            "mcp_and_agentic_skills",
            "coding_and_algorithms",
            "python_software_engineering",
            "mmlu_history_and_systems",
            "biology_genetics_medical_science",
            "nature_ecology_earth_science",
            "tech_history_places_coding_documentation",
            "complex_mathematical_reasoning",
            "scientific_research_and_reasoning",
            "magicoder_oss_instruct",
            "agentic_function_calling",
            "pubmed_clinical_qa",
            "metamath_advanced_proofs",
        ),
        keywords=(),
        cot_samples=(
            "<thought>1. Identify quantities. 2. Apply operation. 3. Verify units.</thought> "
            "If a train travels 120 km in 2 hours, its average speed is 60 km/h.",
            "<thought>1. Recall historical context. 2. Verify chronology.</thought> "
            "In classical antiquity, Helios was the Greek Titan personification of the Sun.",
            "<thought>1. Parse function signature. 2. Check base case.</thought> "
            "def factorial(n: int) -> int: return 1 if n <= 1 else n * factorial(n - 1)",
        ),
    ),
    "fineweb": CategorySpec(
        key="fineweb",
        display_name="FineWeb & Encyclopedic Web",
        description="HuggingFaceFW/fineweb, FineWeb-Edu, DCLM-Baseline, Cosmopedia v2, Wikipedia & Technical Docs",
        sot_prefixes=("1.", "2.", "13."),
        corpus_breakdown_keys=(
            "core_omni_domain_knowledge",
            "tech_history_places_coding_documentation",
        ),
        keywords=(
            "fineweb",
            "wikipedia",
            "encyclopedia",
            "cosmopedia",
            "web",
            "system",
            "cloud",
            "rfc",
        ),
        cot_samples=(
            "<thought>1. Extract main topic. 2. Synthesize encyclopedic facts.</thought> "
            "FineWeb is a 15-trillion-token cleaned and deduplicated English web dataset from CommonCrawl.",
            "<thought>1. Analyze distributed systems architecture. 2. Summarize protocol guarantees.</thought> "
            "HTTP/3 operates over QUIC, multiplexing streams over UDP to eliminate TCP head-of-line blocking.",
        ),
    ),
    "math": CategorySpec(
        key="math",
        display_name="Complex Mathematics & Proofs",
        description="GSM8K, MetaMathQA,hendrycks MATH, ProofNet, NuminaMath & Formal Deductive Logic",
        sot_prefixes=("5.",),
        corpus_breakdown_keys=("complex_mathematical_reasoning", "metamath_advanced_proofs"),
        keywords=(
            "math",
            "equation",
            "theorem",
            "proof",
            "calculus",
            "algebra",
            "integer",
            "prime",
            "solve",
            "number",
        ),
        cot_samples=(
            "<thought>1. Set up linear equation 3x + 7 = 22. 2. Subtract 7: 3x = 15. 3. Divide by 3: x = 5.</thought> "
            "The solution to 3x + 7 = 22 is x = 5.",
            "<thought>1. Apply derivative power rule d/dx(x^n) = n*x^(n-1). 2. Differentiate term by term.</thought> "
            "The derivative of f(x) = 4x^3 - 2x + 9 is f'(x) = 12x^2 - 2.",
        ),
    ),
    "history": CategorySpec(
        key="history",
        display_name="World History & Ancient Civilizations",
        description="World History Timelines, Classical Antiquity, Archaeology, MMLU History & Philippine Heritage",
        sot_prefixes=("3.", "4."),
        corpus_breakdown_keys=("mmlu_history_and_systems",),
        keywords=(
            "history",
            "ancient",
            "empire",
            "century",
            "civilization",
            "war",
            "dynasty",
            "republic",
            "historical",
            "antiquity",
        ),
        cot_samples=(
            "<thought>1. Locate period in classical history. 2. Identify cultural significance.</thought> "
            "The Roman Republic transitioned into the Roman Empire in 27 BCE when Octavian was granted the title Augustus.",
            "<thought>1. Trace Philippine historical chronology. 2. Verify constitutional milestone.</thought> "
            "The Malolos Republic was inaugurated on January 23, 1899 in Barasoain Church, Bulacan.",
        ),
    ),
    "coding": CategorySpec(
        key="coding",
        display_name="Coding, Algorithms & Software Engineering",
        description="The Stack v2, StarCoderData, Magicoder, CodeAlpaca, Python SWE & GitHub Pull Requests",
        sot_prefixes=("9.", "10."),
        corpus_breakdown_keys=(
            "coding_and_algorithms",
            "python_software_engineering",
            "magicoder_oss_instruct",
        ),
        keywords=(
            "def ",
            "class ",
            "import ",
            "python",
            "algorithm",
            "function",
            "code",
            "typescript",
            "rust",
            "complexity",
        ),
        cot_samples=(
            "<thought>1. Use binary search on sorted array. 2. Maintain low and high pointers in O(log n) time.</thought> "
            "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1",
        ),
    ),
    "science": CategorySpec(
        key="science",
        display_name="Science, Biology, Medicine & Ecology",
        description="SciQ, PubMedQA, MedMCQA, PeS2o, Biology, Genetics, Nature & Earth Sciences",
        sot_prefixes=("6.", "7.", "14."),
        corpus_breakdown_keys=(
            "biology_genetics_medical_science",
            "nature_ecology_earth_science",
            "scientific_research_and_reasoning",
            "pubmed_clinical_qa",
        ),
        keywords=(
            "science",
            "biology",
            "cell",
            "dna",
            "protein",
            "clinical",
            "medical",
            "physics",
            "chemistry",
            "ecology",
            "planet",
        ),
        cot_samples=(
            "<thought>1. Trace cellular respiration pathway. 2. Quantify ATP yield in mitochondria.</thought> "
            "Mitochondria generate ATP through oxidative phosphorylation across the inner mitochondrial membrane.",
        ),
    ),
    "geography": CategorySpec(
        key="geography",
        display_name="Places, World Geography & Landmarks",
        description="World Capitals, Topographic Regions, Philippine 17 Administrative Regions & Global Landmarks",
        sot_prefixes=("8.",),
        corpus_breakdown_keys=("core_omni_domain_knowledge",),
        keywords=(
            "geography",
            "capital",
            "continent",
            "ocean",
            "mountain",
            "river",
            "region",
            "province",
            "country",
            "island",
        ),
        cot_samples=(
            "<thought>1. Identify archipelago geography. 2. Verify regional capital.</thought> "
            "The Philippines is an archipelago of 7,641 islands divided into Luzon, Visayas, and Mindanao, with Manila as its capital.",
        ),
    ),
    "law": CategorySpec(
        key="law",
        display_name="Law, Governance, Philosophy & Ethics",
        description="Constitutional Law, Jurisprudence, Statutory Interpretation, Formal Logic & Epistemic Ethics",
        sot_prefixes=("15.",),
        corpus_breakdown_keys=("core_omni_domain_knowledge",),
        keywords=(
            "law",
            "legal",
            "constitution",
            "statute",
            "philosophy",
            "ethics",
            "rights",
            "court",
            "justice",
            "governance",
        ),
        cot_samples=(
            "<thought>1. Examine constitutional separation of powers. 2. Check legislative, executive, and judicial branches.</thought> "
            "Constitutional democracies divide sovereign authority among legislative, executive, and judicial branches to maintain checks and balances.",
        ),
    ),
    "agents": CategorySpec(
        key="agents",
        display_name="Agentic Skills & Model Context Protocol (MCP)",
        description="BFCL Function Calling, JSON-RPC Tool Invocations, Multi-Step Planning & Autonomous Recovery",
        sot_prefixes=("11.", "12."),
        corpus_breakdown_keys=("mcp_and_agentic_skills", "agentic_function_calling"),
        keywords=(
            "tool_call",
            "mcp",
            "json-rpc",
            "agent",
            "function",
            "schema",
            "api",
            "parameter",
        ),
        cot_samples=(
            "<thought>1. Match user intent to registered MCP tool schema. 2. Emit structured JSON-RPC arguments.</thought> "
            '<tool_call>{"name": "calculator", "arguments": {"expression": "144 * 25"}}</tool_call>',
        ),
    ),
    "tagalog": CategorySpec(
        key="tagalog",
        display_name="Bilingual Filipino / Tagalog & Multilingual",
        description="Tagalog Grammar, Bilingual Q&A, Philippine Cultural Grounding & Cross-Lingual Translation",
        sot_prefixes=("16.",),
        corpus_breakdown_keys=("core_omni_domain_knowledge",),
        keywords=(
            "tagalog",
            "filipino",
            "ang ",
            "mga ",
            "sa ",
            "ito",
            "pilipinas",
            "kamusta",
            "sino",
            "ano",
        ),
        cot_samples=(
            "<thought>1. Unawain ang tanong sa wikang Tagalog. 2. Magbigay ng malinaw at tumpak na sagot.</thought> "
            "Ang Helios ay isang bukas na modelo ng wika (open-weight LLM) na nilikha ni Arron Kian Parejas sa Pilipinas.",
        ),
    ),
}

CATEGORY_ALIASES: dict[str, str] = {
    "omni": "all",
    "default": "all",
    "web": "fineweb",
    "encyclopedia": "fineweb",
    "mathematics": "math",
    "hist": "history",
    "code": "coding",
    "swe": "coding",
    "python": "coding",
    "bio": "science",
    "medical": "science",
    "medicine": "science",
    "nature": "science",
    "geo": "geography",
    "places": "geography",
    "legal": "law",
    "philosophy": "law",
    "mcp": "agents",
    "agent": "agents",
    "tools": "agents",
    "filipino": "tagalog",
    "ph": "tagalog",
}


def normalize_category(category: str | None) -> str:
    """Resolve user-supplied category string or alias to a canonical key."""
    if not category:
        return "all"
    cleaned = category.strip().lower()
    if cleaned in CATEGORY_SPECS:
        return cleaned
    if cleaned in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[cleaned]
    valid = ", ".join(CATEGORY_SPECS.keys())
    raise ValueError(f"Unknown pretraining category '{category}'. Valid categories: {valid}")


def matches_category(record: dict[str, Any], category: str) -> bool:
    """Return True if a JSONL corpus record belongs to the target category."""
    cat_key = normalize_category(category)
    if cat_key == "all":
        return True
    spec = CATEGORY_SPECS[cat_key]
    meta = " ".join(
        str(record.get(k, "")) for k in ("domain", "category", "source", "id") if record.get(k)
    ).lower()
    if any(kw in meta for kw in spec.keywords):
        return True
    text = (
        record.get("text") or f"{record.get('prompt', '')} {record.get('response', '')}"
    ).lower()
    return any(kw in text for kw in spec.keywords)


def get_category_catalog(sot_path: Path | str = "sot/datasets_sot.json") -> list[dict[str, Any]]:
    """Return structured summary of all 10 categories with exact dataset and record counts."""
    p = Path(sot_path)
    datasets: list[dict[str, Any]] = []
    breakdown: dict[str, int] = {}
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        datasets = data.get("datasets", [])
        corpora = data.get("consolidated_corpora", [])
        if corpora:
            breakdown = corpora[0].get("breakdown", {})

    catalog: list[dict[str, Any]] = []
    for key, spec in CATEGORY_SPECS.items():
        matched_ds = [
            d
            for d in datasets
            if any(d.get("category", "").startswith(pref) for pref in spec.sot_prefixes)
        ]
        records_count = sum(breakdown.get(bk, 0) for bk in spec.corpus_breakdown_keys)
        catalog.append(
            {
                "key": key,
                "display_name": spec.display_name,
                "description": spec.description,
                "dataset_count": len(matched_ds),
                "dataset_names": [d["name"] for d in matched_ds],
                "records_count": records_count,
                "train_cmd": f"/train all --category {key} --minutes 20",
            }
        )
    return catalog
