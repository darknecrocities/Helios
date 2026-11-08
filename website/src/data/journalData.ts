import type { DocChapter } from '../types/models';

export const JOURNAL_AND_RUNTIME_CHAPTERS: DocChapter[] = [
  {
    id: 'journal',
    romanNumeral: '01',
    title: "Architect's Engineering Journal (Research → Problem → Solution)",
    subtitle:
      'A complete engineering walkthrough by Arron Kian Parejas on how Helios was researched, architected, and built from first principles in the Philippines.',
    greekTerm: 'Engineering Journal',
    summary:
      'Helios was conceived, architected, and engineered from the ground up by Arron Kian Parejas in the Philippines. This engineering journal documents the exact research questions, architectural bottlenecks, and mathematical solutions behind the 6-tier Helios ecosystem (Nano 17.84M to Flagship 5.80B).',
    sections: [
      {
        heading: 'Entry I · Epistemic Honesty vs. Fluent Hallucination (Research → Problem → Solution)',
        body: `• Research Observation: Standard autoregressive transformers optimize purely for next-token cross-entropy. When queried about obscure historical dates, biographical facts, or out-of-distribution entities, they generate syntactically confident falsehoods rather than admitting uncertainty.
• The Engineering Problem: How do we guarantee that a local foundation model answers with verifiable facts when grounded knowledge exists, performs exact arithmetic without guessing, and explicitly abstains when a topic lies outside its verified knowledge boundary?
• The Helios Solution: We engineered a deterministic 5-stage inference pipeline inside helios/chat/responder.py: (1) Query Normalization & Bilingual Intent Detection (English + Filipino/Tagalog), (2) Deterministic AST Mathematical & Unit Evaluation, (3) Subject-Aware Semantic Verification (verify_subject_overlap) that checks entity overlap before accepting any retrieved passage, and (4) Zero-Hallucination Epistemic Fallback that cleanly states when a query is outside verified scope.`,
        formula:
          'Verified(d, q) = (SubjectOverlap(Entities(q), Entities(d)) ≥ τ_subj) ∧ (RRF(d, q) ≥ τ_conf)',
      },
      {
        heading: 'Entry II · Breaking the 256K Context Memory Wall on 16GB Laptops',
        body: `• Research Observation: In standard Multi-Head Attention (MHA), scaling the context window to 262,144 tokens (256K) on a 6B-class model requires 34.36 GB of FP16 KV-Cache RAM alone—instantly crashing consumer laptops.
• The Engineering Problem: How can Helios Sol Invictus (5.80B Flagship) process 256K tokens of context while fitting comfortably inside a standard 16GB RAM MacBook or workstation?
• The Helios Solution: First, we replaced MHA with Grouped-Query Attention (4:1 and 3:1 GQA ratios), immediately cutting KV-Cache footprint by 4×. Second, we implemented Dynamic NTK High-Theta Rotary Position Embeddings (θ scaled from 100,000 on Nano up to 10,000,000 on Flagship) so long-range attention never degrades. Third, we built a pre-allocated Contiguous KV-Cache with optional symmetric INT8 quantization (s = max(|x|)/127), compressing the 256K Flagship KV-Cache from 34.36 GB down to 8.59 GB.`,
        formula:
          'KV_RAM(Flagship, 256K, INT8) = 2 × L(32) × H_kv(8) × d_head(128) × T(262,144) × 1 byte = 8.59 GB',
      },
      {
        heading: 'Entry III · Lossless Bilingual (English + Tagalog) & Source Code Tokenization',
        body: `• Research Observation: Off-the-shelf English tokenizers fragment Filipino/Tagalog affixes (mag-, nag-, -in-, -um-) and multi-space Python/TypeScript indentation into inefficient single-character tokens, inflating sequence length and slowing generation.
• The Engineering Problem: Design a compact 32,000-token vocabulary that guarantees zero <unk> out-of-vocabulary failures across arbitrary UTF-8 streams while maintaining high token fertility across English, Tagalog, and source code.
• The Helios Solution: We built two production tokenizers from scratch in helios/tokenizer/: a Byte-Level BPE Tokenizer (256 base byte tokens + regex pre-tokenization + priority merge rules) and a Viterbi Unigram Language Model Tokenizer trained via Expectation-Maximization (EM) with Digamma log-probabilities. Both reserve dedicated control tokens for ChatML, Chain-of-Thought (<|thought|>), MCP tool calls (<|call:tool|>), and multimodal vision (<|image|>).`,
      },
      {
        heading: 'Entry IV · 2.4× Speculative Decoding (Pairing Nano-20M with Flagship-5.80B)',
        body: `• Research Observation: Autoregressive generation on a 5.80B parameter model is memory-bandwidth bound because every single generated token requires reading 2.84 GB of weights from RAM.
• The Engineering Problem: How do we double Flagship token throughput without sacrificing a single bit of mathematical or reasoning accuracy?
• The Helios Solution: Because all six Helios tiers share the exact same 32,000-token vocabulary and tokenizer, Helios Aether (NANO · 17.84M) serves as an ultra-fast speculative draft model (γ = 4 lookahead tokens at 265 tok/s) while Helios Sol Invictus (FLAGSHIP · 5.80B) verifies all drafted tokens in a single parallel forward pass using Leviathan et al. (2023) rejection sampling. This delivers a verified 2.4× wall-clock speedup (53.3 tok/s vs 22.2 tok/s) with an identical output distribution.`,
        formula:
          'Accept(x_i) = min(1, p_target(x_i | x_{<i}) / q_draft(x_i | x_{<i}))',
      },
      {
        heading: 'Entry V · Two-Tier Hybrid FTS5 + FAISS Memory & Zero-Hang Live Switching',
        body: `• Research Observation: Loading 134,728 grounded records (402 MB JSONL) into an in-memory BM25 index at CLI startup caused multi-second boot delays, while switching between 1B, 3B, and 6B models mid-session risked Unified Memory OOM hangs.
• The Engineering Problem: Achieve <10ms retrieval over 400MB+ of grounded knowledge with instant startup, and allow live mid-conversation switching across all 6 tiers without memory leaks.
• The Helios Solution: We engineered a Two-Tier Hybrid Semantic Store combining a <1ms 15K in-memory Okapi BM25 index with an On-Disk SQLite FTS5 inverted index (134,728 records queried in <10ms) and Dense FAISS vectors unified via Reciprocal Rank Fusion (RRF). For live model switching (/switch), we built a transactional pre-flight RAM guard using psutil + torch.mps.synchronize() that validates available memory before weight allocation and automatically rolls back on failure.`,
      },
    ],
  },
  {
    id: 'quickstart',
    romanNumeral: '02',
    title: 'How to Run Helios (Ollama, Hugging Face & CLI)',
    subtitle:
      'Official registry commands for parejasarronkian/Helios on Ollama, Darknecrocities/Helios on Hugging Face, and Python CLI.',
    greekTerm: 'Installation & Runtime',
    summary:
      'Every Helios model is published under the official Ollama registry (parejasarronkian/Helios) and Hugging Face Hub (Darknecrocities/Helios) in GGUF, SafeTensors, and ONNX formats.',
    sections: [
      {
        heading: '1. Running via Official Ollama Registry (parejasarronkian/Helios)',
        body: 'Pull and run any of the six Helios tiers directly from https://ollama.com/parejasarronkian on macOS, Linux, or Windows:',
        codeSnippet: {
          language: 'bash',
          label: 'Terminal · Official Ollama Registry Commands',
          code: `# Run the 5.80B Flagship (Helios Sol Invictus · 256K Context)
ollama run parejasarronkian/Helios:flagship
ollama run parejasarronkian/Helios:latest

# Or run specific tiers matched to your hardware (NANO to PRO):
ollama run parejasarronkian/Helios:nano       # Aether (17.84M · 61.3 MB RAM · 16K Ctx)
ollama run parejasarronkian/Helios:alpha      # Phaethon (100.09M · 138.1 MB RAM · 32K Ctx)
ollama run parejasarronkian/Helios:beta       # Eos (303.35M · 295.9 MB RAM · 64K Ctx)
ollama run parejasarronkian/Helios:core       # Hyperion (1.13B · 821.2 MB RAM · 128K Ctx)
ollama run parejasarronkian/Helios:pro        # Apollo (3.32B · 2.17 GB RAM · 128K Ctx)`,
        },
      },
      {
        heading: '2. Downloading from Hugging Face (Darknecrocities/Helios)',
        body: 'Download full SafeTensors weights, BPE/Unigram tokenizers, LoRA adapters, and GGUF quantizations from https://huggingface.co/Darknecrocities/Helios:',
        codeSnippet: {
          language: 'bash',
          label: 'Terminal · Hugging Face CLI & Python SDK',
          code: `# Download the complete repository or a specific tier checkpoint
huggingface-cli download Darknecrocities/Helios --local-dir ./checkpoints/helios

# Or load directly in Python with HeliosForCausalLM:
uv run python -c "
from helios.model.config import HeliosConfig
from helios.model.transformer import HeliosForCausalLM
cfg = HeliosConfig.from_preset('flagship-6b')
model = HeliosForCausalLM(cfg)
print(model.summary())
"`,
        },
      },
      {
        heading: '3. Interactive Native CLI & Live Model Switcher',
        body: 'Launch the native Helios interactive shell with Chain-of-Thought visualization, 4-Tier Hierarchical Memory, and zero-hang live switching across all 6 tiers:',
        codeSnippet: {
          language: 'bash',
          label: 'Terminal · Native Helios Interactive REPL',
          code: `# Launch interactive chat on Helios Hyperion (CORE · 1.13B)
uv run python -m helios.cli chat --preset core-1b

# Built-in Interactive Slash Commands inside the session:
# /model              -> Inspect active tier parameters, layers, GQA ratio, and RAM
# /switch flagship-6b -> Live-switch to Sol Invictus (5.80B) with pre-flight RAM check
# /think on           -> Toggle visible Chain-of-Thought (<|thought|>) reasoning trace
# /memory             -> Inspect Working, Episodic WAL, and FTS5 Semantic Memory hits
# /stats              -> View real-time tok/s throughput and KV-cache compression`,
        },
      },
    ],
  },
  {
    id: 'mcp-memory',
    romanNumeral: '03',
    title: 'Model Context Protocol (MCP) & 4-Tier Memory (HMA)',
    subtitle:
      'Native JSON-RPC 2.0 tool execution, Episodic SQLite WAL persistence, and Two-Tier FTS5 + FAISS hybrid retrieval.',
    greekTerm: 'MCP & Memory Architecture',
    summary:
      'Helios integrates a biological 4-Tier Hierarchical Memory Architecture (HMA) and native Model Context Protocol (MCP) JSON-RPC 2.0 tool calling directly into its token vocabulary and runtime.',
    sections: [
      {
        heading: '1. Four-Tier Hierarchical Memory Architecture (HMA)',
        body: 'Unlike stateless LLMs that forget everything once the context window rolls over, Helios coordinates four specialized memory layers:',
        metricsTable: {
          headers: ['Memory Tier', 'Storage Engine', 'Latency', 'Capacity', 'Primary Responsibility'],
          rows: [
            ['Tier 1 · Working Memory', 'In-RAM Sliding Token Buffer', '< 0.1 ms', '16K – 256K Tokens', 'Active multi-turn conversation state & automatic context compaction'],
            ['Tier 2 · Episodic Memory', 'SQLite 3 (WAL Mode + ACID)', '< 2.0 ms', 'Unlimited Sessions', 'Cross-session chronological dialogue recall & user preference persistence'],
            ['Tier 3 · Semantic Memory', '15K RAM BM25 + SQLite FTS5 + FAISS', '< 10.0 ms', '134,728+ Docs (402 MB)', 'Bilingual Reciprocal Rank Fusion (RRF) over grounded encyclopedic & code corpora'],
            ['Tier 4 · Procedural Memory', 'MCP JSON-RPC 2.0 + AST Evaluator', 'Deterministic', 'Extensible Tool Registry', 'Schema-validated external tool calls, calculator, unit conversion & code sandbox'],
          ],
        },
      },
      {
        heading: '2. Native Model Context Protocol (MCP) Tool-Calling Lifecycle',
        body: 'Helios encodes tool invocations using dedicated special tokens (<|call:tool|>...<|end_call|>) and parses tool responses via (<|return:tool|>...<|end_return|>). Every tool call is validated against strict JSON Schema definitions before execution.',
        codeSnippet: {
          language: 'python',
          label: 'Python · Registering & Executing an MCP Tool in Helios',
          code: `from helios.memory.mcp import MCPToolRegistry, MCPToolSpec

registry = MCPToolRegistry()
registry.register(
    MCPToolSpec(
        name="query_civic_budget",
        description="Retrieve verified municipal budget allocations by fiscal year.",
        parameters={
            "type": "object",
            "properties": {
                "fiscal_year": {"type": "integer"},
                "department": {"type": "string"}
            },
            "required": ["fiscal_year", "department"]
        },
        handler=lambda args: {"allocated_php": 145000000, "status": "audited"}
    )
)

# Model emits: <|call:tool|>{"name":"query_civic_budget","arguments":{"fiscal_year":2026,"department":"Health"}}<|end_call|>
result = registry. invoke_from_token_stream(raw_model_output)`,
        },
      },
    ],
  },
];
