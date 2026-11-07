"""Multilingual and code fertility evaluation harness for Helios tokenizers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HeliosTokenizerProtocol(Protocol):
    """Protocol matching the HeliosTokenizer interface contract."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        allowed_special: Any = "all",
    ) -> list[int]:
        """Convert input text string into a list of token IDs."""
        ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """Convert token ID list back to string."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        ...


@dataclass(frozen=True)
class FertilityMetrics:
    """Aggregated fertility and throughput metrics for a benchmark text suite."""

    domain: str
    bytes_per_token: float
    tokens_per_word: float
    tokens_per_byte: float
    chars_per_token: float
    total_tokens: int
    total_words: int
    total_bytes: int
    total_chars: int
    unique_tokens: int
    vocab_coverage_ratio: float
    elapsed_seconds: float
    tokens_per_second: float


@dataclass
class ComparisonReport:
    """Comparative report evaluating multiple tokenizers across benchmark domains."""

    tokenizer_names: list[str]
    domains: list[str]
    metrics_by_tokenizer: dict[str, dict[str, FertilityMetrics]] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render report as a GitHub-flavored Markdown table."""
        lines = [
            "# Helios Tokenizer Fertility Comparison Report",
            "",
            "| Domain | Metric | " + " | ".join(self.tokenizer_names) + " | Delta (%) |",
            "|---|---|" + "|".join("---" for _ in self.tokenizer_names) + "|---|",
        ]
        metrics_keys = [
            ("Bytes / Token", "bytes_per_token", "{:.2f}"),
            ("Tokens / Word", "tokens_per_word", "{:.2f}"),
            ("Tokens / Byte", "tokens_per_byte", "{:.3f}"),
            ("Chars / Token", "chars_per_token", "{:.2f}"),
            ("Tokens / Sec", "tokens_per_second", "{:,.0f}"),
        ]

        for domain in self.domains:
            for label, attr, fmt in metrics_keys:
                row_vals: list[str] = []
                for name in self.tokenizer_names:
                    m = self.metrics_by_tokenizer[name][domain]
                    val = getattr(m, attr)
                    row_vals.append(fmt.format(val))

                delta_str = "N/A"
                if len(self.tokenizer_names) >= 2:
                    v0 = getattr(self.metrics_by_tokenizer[self.tokenizer_names[0]][domain], attr)
                    v1 = getattr(self.metrics_by_tokenizer[self.tokenizer_names[1]][domain], attr)
                    if v0 > 0:
                        delta = ((v1 - v0) / v0) * 100.0
                        delta_str = f"{delta:+.1f}%"

                row_str = " | ".join(row_vals)
                lines.append(f"| {domain} | {label} | {row_str} | {delta_str} |")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert comparative report to dictionary format for JSON export."""
        out: dict[str, Any] = {
            "tokenizers": self.tokenizer_names,
            "domains": self.domains,
            "results": {},
        }
        for t_name, domains_dict in self.metrics_by_tokenizer.items():
            out["results"][t_name] = {}
            for d_name, m in domains_dict.items():
                out["results"][t_name][d_name] = {
                    "bytes_per_token": m.bytes_per_token,
                    "tokens_per_word": m.tokens_per_word,
                    "tokens_per_byte": m.tokens_per_byte,
                    "chars_per_token": m.chars_per_token,
                    "total_tokens": m.total_tokens,
                    "total_words": m.total_words,
                    "total_bytes": m.total_bytes,
                    "total_chars": m.total_chars,
                    "unique_tokens": m.unique_tokens,
                    "vocab_coverage_ratio": m.vocab_coverage_ratio,
                    "elapsed_seconds": m.elapsed_seconds,
                    "tokens_per_second": m.tokens_per_second,
                }
        return out


# Curated representative benchmark text suites (zero mock data)
BENCHMARK_SUITES: dict[str, list[str]] = {
    "english_literature": [
        (
            "It is a truth universally acknowledged, that a single man in possession "
            "of a good fortune, must be in want of a wife. However little known the feelings "
            "or views of such a man may be on his first entering a neighbourhood, this "
            "truth is so well fixed in the minds of the surrounding families, that he is "
            "considered the rightful property of some one or other of their daughters."
        ),
        (
            "The past is a foreign country; they do things differently there. Memory believes "
            "before knowing remembers. Believes longer than recollects, longer than knowing even "
            "wonders. No man ever steps in the same river twice, for it is not the same river "
            "and he is not the same man."
        ),
    ],
    "english_technical": [
        (
            "The Helios architecture utilizes Grouped-Query Attention (GQA) where key and value "
            "projections are grouped by a factor of three relative to query heads. To prevent "
            "catastrophic forgetting and KV-cache bloat, dynamic episodic memory is decoupled "
            "into a persistent SQLite store operating in Write-Ahead Logging mode, while base "
            "Transformer weights remain immutable during inference."
        ),
        (
            "Numerically stable Pre-RMSNorm normalization applies root-mean-square scaling "
            "prior to each attention and feed-forward layer. Position embeddings employ Rotary "
            "Position Embeddings (RoPE) with precomputed trigonometric frequency caches and "
            "configurable base frequency scaling."
        ),
    ],
    "filipino_conversational": [
        (
            "Magandang umaga sa inyong lahat! Kumusta naman ang inyong naging araw? Sana ay naging "
            "maayos at mabunga ang inyong pagpupursigi sa trabaho at pag-aaral ngayong linggo. "
            "Saan nga pala tayo magkikita mamayang hapon para sa paghahanda ng ating proyekto? "
            "Pakisabi na lang kay Nanay na medyo mahuhuli ako ng kaunti dahil sa trapiko sa EDSA."
        ),
        (
            "Napakasarap kumain ng mainit na sinigang habang umuulan sa labas. Huwag mong "
            "kalimutang magdala ng payong kapag ikaw ay aalis mamaya upang hindi ka magkasakit."
        ),
    ],
    "filipino_morphological": [
        (
            "Ang taong nagpapakapagod at pinagkakatiwalaan ng sambayanan ay patuloy na "
            "nakikipagkapwa-tao kahit gaano man kabigat ang kanyang mga pasanin at pananagutan. "
            "Ipagpatawad po ninyo ang aming naging pagkukulang; ikinagagalak naming ipagbigay-alam "
            "na magpapakadalubhasa ang ating mga kawani upang mapagbuti ang pampublikong serbisyo."
        ),
        (
            "Napag-alaman ng pamunuan na kinakailangang magtulungan ang bawat isa sa pamamagitan "
            "ng pakikipagtalastasan at pagpapahalaga sa kapakanan ng nakararami."
        ),
    ],
    "code_python": [
        (
            "import asyncio\n"
            "from dataclasses import dataclass\n"
            "from typing import Any, Optional\n\n"
            "@dataclass\n"
            "class CacheNode:\n"
            "    key: str\n"
            "    value: Any\n"
            "    ttl_seconds: float = 300.0\n\n"
            "async def fetch_and_cache(entry_id: str, client: Any) -> Optional[CacheNode]:\n"
            "    try:\n"
            "        raw_data = await client.get_resource(entry_id)\n"
            "        if not raw_data:\n"
            "            return None\n"
            "        return CacheNode(key=entry_id, value=raw_data['payload'])\n"
            "    except (ConnectionError, TimeoutError) as exc:\n"
            "        raise RuntimeError(f'Failed to fetch {entry_id}: {exc}') from exc\n"
        ),
    ],
    "code_rust": [
        (
            "pub enum TokenizerResult<T> {\n"
            "    Success(T),\n"
            "    Error(String),\n"
            "}\n\n"
            "pub trait Tokenizer {\n"
            "    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;\n"
            "    fn decode(&self, ids: &[u32]) -> Result<String, String>;\n"
            "}\n\n"
            "impl Tokenizer for ByteBPE {\n"
            "    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {\n"
            "        let bytes = text.as_bytes();\n"
            "        let mut tokens = Vec::with_capacity(bytes.len());\n"
            "        for &b in bytes {\n"
            "            tokens.push(b as u32);\n"
            "        }\n"
            "        Ok(tokens)\n"
            "    }\n"
            "}\n"
        ),
    ],
    "code_javascript": [
        (
            "async function streamTokens(prompt, modelEndpoint) {\n"
            "  const response = await fetch(`${modelEndpoint}/v1/completions`, {\n"
            "    method: 'POST',\n"
            "    headers: { 'Content-Type': 'application/json' },\n"
            "    body: JSON.stringify({ prompt, max_tokens: 512, stream: true })\n"
            "  });\n"
            "  if (!response.ok) {\n"
            "    throw new Error(`HTTP error! status: ${response.status}`);\n"
            "  }\n"
            "  const reader = response.body.getReader();\n"
            "  return reader;\n"
            "}\n"
        ),
    ],
    "code_sql": [
        (
            "WITH RankedInteractions AS (\n"
            "    SELECT\n"
            "        session_id,\n"
            "        user_id,\n"
            "        prompt_tokens,\n"
            "        completion_tokens,\n"
            "        created_at,\n"
            "        ROW_NUMBER() OVER "
            "        (PARTITION BY user_id ORDER BY created_at DESC) as rank_order\n"
            "    FROM model_audit_log\n"
            "    WHERE created_at >= NOW() - INTERVAL '30 days'\n"
            ")\n"
            "SELECT\n"
            "    user_id,\n"
            "    COUNT(session_id) AS total_sessions,\n"
            "    SUM(prompt_tokens + completion_tokens) AS total_tokens_consumed,\n"
            "    AVG(completion_tokens) AS avg_completion_length\n"
            "FROM RankedInteractions\n"
            "WHERE rank_order <= 100\n"
            "GROUP BY user_id\n"
            "HAVING COUNT(session_id) > 5\n"
            "ORDER BY total_tokens_consumed DESC;\n"
        ),
    ],
}


def calculate_fertility_metrics(
    tokenizer: HeliosTokenizerProtocol,
    texts: list[str],
    domain: str = "general",
) -> FertilityMetrics:
    """Calculate fertility and throughput metrics for a tokenizer over a suite of texts."""
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    total_chars = 0
    unique_tokens_set: set[int] = set()

    start_time = time.perf_counter()
    for text in texts:
        utf8_bytes = text.encode("utf-8")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        words = [w for w in text.split() if w]

        total_bytes += len(utf8_bytes)
        total_tokens += len(tokens)
        total_words += len(words)
        total_chars += len(text)
        unique_tokens_set.update(tokens)

    elapsed = max(time.perf_counter() - start_time, 1e-6)

    bpt = (total_bytes / total_tokens) if total_tokens > 0 else 0.0
    tpw = (total_tokens / total_words) if total_words > 0 else 0.0
    tpb = (total_tokens / total_bytes) if total_bytes > 0 else 0.0
    cpt = (total_chars / total_tokens) if total_tokens > 0 else 0.0
    v_size = max(tokenizer.vocab_size, 1)
    vocab_cov = len(unique_tokens_set) / v_size
    tps = total_tokens / elapsed

    return FertilityMetrics(
        domain=domain,
        bytes_per_token=bpt,
        tokens_per_word=tpw,
        tokens_per_byte=tpb,
        chars_per_token=cpt,
        total_tokens=total_tokens,
        total_words=total_words,
        total_bytes=total_bytes,
        total_chars=total_chars,
        unique_tokens=len(unique_tokens_set),
        vocab_coverage_ratio=vocab_cov,
        elapsed_seconds=elapsed,
        tokens_per_second=tps,
    )


def compare_tokenizers(
    tokenizers: dict[str, HeliosTokenizerProtocol],
    suites: dict[str, list[str]] | None = None,
) -> ComparisonReport:
    """Run full comparative fertility evaluation across tokenizers and benchmark domains."""
    if suites is None:
        suites = BENCHMARK_SUITES

    report = ComparisonReport(
        tokenizer_names=list(tokenizers.keys()),
        domains=list(suites.keys()),
    )

    for t_name, tokenizer in tokenizers.items():
        report.metrics_by_tokenizer[t_name] = {}
        for d_name, texts in suites.items():
            metrics = calculate_fertility_metrics(tokenizer, texts, domain=d_name)
            report.metrics_by_tokenizer[t_name][d_name] = metrics

    return report
