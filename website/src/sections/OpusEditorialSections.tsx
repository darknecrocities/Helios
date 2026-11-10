import React from 'react';
import pretrainData from '../data/pretrain_metrics.json';

interface MarginaliaItem {
  label: string;
  value: string;
}

interface EditorialBlockProps {
  id: string;
  numeral: string;
  category: string;
  title: string;
  leadParagraph: string;
  paragraphs: string[];
  marginalia: MarginaliaItem[];
  quote?: {
    text: string;
    attribution: string;
  };
  specGrid?: {
    heading: string;
    detail: string;
  }[];
  sideVideo?: {
    src: string;
    caption: string;
    badge: string;
  };
  bgVariant?: 'cream' | 'sand';
}

const EditorialNarrativeSection: React.FC<EditorialBlockProps> = ({
  id,
  numeral,
  category,
  title,
  leadParagraph,
  paragraphs,
  marginalia,
  quote,
  specGrid,
  sideVideo,
  bgVariant = 'cream',
}) => {
  const bgClass = bgVariant === 'sand' ? 'bg-[#F4EFE6]' : 'bg-[#FBF9F5]';

  return (
    <section
      id={id}
      className={`relative py-20 md:py-28 ${bgClass} text-[#141312] border-t border-[#141312]/10`}
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-16 items-start">
          {/* Left Sticky Column (4 cols): Claude Opus–style Editorial Index & Marginalia */}
          <div className="lg:col-span-4 lg:sticky lg:top-28 space-y-6">
            <div className="flex items-baseline gap-3 border-b border-[#141312]/15 pb-3">
              <span className="font-serif-editorial italic text-2xl sm:text-3xl text-[#FF5500] font-semibold">
                {numeral}.
              </span>
              <span className="font-mono-tabular text-[11px] uppercase tracking-[0.2em] text-[#141312]/60">
                {category}
              </span>
            </div>

            <h2 className="font-display text-2xl sm:text-3xl lg:text-[34px] font-bold tracking-tight text-[#141312] leading-[1.14]">
              {title}
            </h2>

            {quote && (
              <blockquote className="pt-2 border-l-2 border-[#FF5500] pl-4 space-y-1.5">
                <p className="font-serif-editorial italic text-base sm:text-lg text-[#141312]/80 leading-relaxed">
                  “{quote.text}”
                </p>
                <footer className="font-mono-tabular text-[10px] uppercase tracking-wider text-[#141312]/50">
                  — {quote.attribution}
                </footer>
              </blockquote>
            )}

            {/* Key Architectural Marginalia */}
            <div className="pt-4 border-t border-[#141312]/10 space-y-3">
              {marginalia.map((item) => (
                <div key={item.label} className="flex items-baseline justify-between gap-4 text-xs">
                  <span className="font-mono-tabular text-[10px] uppercase tracking-wider text-[#141312]/50">
                    {item.label}
                  </span>
                  <span className="font-mono-tabular font-semibold text-[#141312] text-right">
                    {item.value}
                  </span>
                </div>
              ))}
            </div>

            {/* Side In-Situ Video Player Widget */}
            {sideVideo && (
              <div className="pt-4 border-t border-[#141312]/10 space-y-2">
                <div className="rounded-xl overflow-hidden bg-black/90 border border-[#141312]/15 shadow-md aspect-video relative group">
                  <video
                    src={sideVideo.src}
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/60 backdrop-blur-xs text-[9px] font-mono-tabular text-[#FF5500] border border-white/10">
                    {sideVideo.badge}
                  </div>
                </div>
                <p className="text-[11px] font-mono-tabular text-[#141312]/60 leading-normal">
                  {sideVideo.caption}
                </p>
              </div>
            )}
          </div>

          {/* Right Long-Form Reading Column (8 cols): Opus Magazine Prose */}
          <div className="lg:col-span-8 space-y-6 max-w-[68ch]">
            {/* Lead Editorial Serif Paragraph */}
            <p className="font-serif-editorial text-xl sm:text-2xl md:text-[26px] text-[#141312] leading-[1.45] font-normal">
              {leadParagraph}
            </p>

            <div className="h-[1px] w-16 bg-[#FF5500]/60 my-2" />

            {/* Multi-Paragraph Technical Narrative */}
            <div className="space-y-5 text-[15px] sm:text-base text-[#141312]/80 leading-[1.82] font-normal">
              {paragraphs.map((para, idx) => (
                <p key={idx}>{para}</p>
              ))}
            </div>

            {/* Optional Clean Typographic Specification Grid */}
            {specGrid && specGrid.length > 0 && (
              <div className="pt-6 mt-6 border-t border-[#141312]/12 grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-5">
                {specGrid.map((spec) => (
                  <div key={spec.heading} className="space-y-1">
                    <h3 className="font-display text-xs sm:text-sm font-bold uppercase tracking-wider text-[#141312]">
                      {spec.heading}
                    </h3>
                    <p className="text-xs sm:text-[13px] text-[#141312]/70 leading-relaxed">
                      {spec.detail}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export const EditorialReasoningSection: React.FC = () => (
  <EditorialNarrativeSection
    id="editorial-reasoning"
    numeral="I"
    category="Extended Thinking & Autonomy"
    title="How Helios Thinks, Reasons, and Acts Before Speaking"
    bgVariant="sand"
    leadParagraph="Most language models generate responses reflexively, committing to a conclusion token by token before verifying whether the underlying premise holds. Helios is architected to deliberate first—separating internal step-by-step reasoning from its final spoken synthesis."
    paragraphs={[
      "When presented with a multi-step mathematical proof, a complex codebase refactoring task, or an ambiguous scientific query, Helios opens a dedicated internal reasoning workspace delimited by native <|thought|> and <|end_thought|> control tokens. Inside this scratchpad, the model decomposes the problem into verifiable sub-goals, tests edge cases, and audits its own intermediate logic before emitting a single user-facing sentence. Users can inspect this reasoning trace in real time or collapse it for concise answers.",
      "Beyond passive text generation, Helios is built from the ground up as an autonomous agentic engine. Its vocabulary natively encodes Model Context Protocol (MCP) JSON-RPC 2.0 invocations and Nous Research Hermes XML function-calling schemas. Whether orchestrating terminal sandboxes in OpenClaw, executing multi-file repository edits in Google Antigravity, or running deterministic Python AST calculations, Helios treats external tools as first-class extensions of its own cognition.",
      "To make deep deliberation instantaneous on consumer hardware, all six Helios tiers share an identical 32,000-token vocabulary and embedding topology. This allows Helios Aether (Nano, 17.84M parameters) to act as an ultra-fast speculative draft engine—proposing four candidate tokens in under 3 milliseconds—while Helios Sol Invictus (Flagship, 5.80B parameters) verifies them in a single parallel forward pass. The result is a mathematically lossless 2.4× speedup in real-world generation."
    ]}
    quote={{
      text: "True intelligence is measured not by how quickly a model speaks, but by how rigorously it verifies its own chain of thought before answering.",
      attribution: "Helios Architecture Monograph"
    }}
    marginalia={[
      { label: "Reasoning Format", value: "Native <|thought|> CoT" },
      { label: "Agent Protocols", value: "MCP JSON-RPC · OpenClaw · Hermes" },
      { label: "Speculative Tandem", value: "Nano (17.8M) → Flagship (5.8B)" },
      { label: "Verified Speedup", value: "2.4× – 2.8× Lossless" },
    ]}
    sideVideo={{
      src: "/assets/video/helios_ollama_reasoning.mp4",
      badge: "LIVE DELIBERATION",
      caption: "Screen capture of Helios decomposing complex mathematical equations into step-by-step proofs before outputting answers."
    }}
    specGrid={[
      {
        heading: "Deliberative Scratchpad",
        detail: "Dedicated Chain-of-Thought tokens allow the model to backtrack, verify arithmetic, and plan multi-turn tool trajectories prior to final synthesis."
      },
      {
        heading: "Zero-Drift Tool Execution",
        detail: "Strict JSON Schema validation ensures every OpenClaw, Hermes, and Antigravity MCP tool call is syntactically sound and deterministic."
      }
    ]}
  />
);

export const EditorialTrainingSection: React.FC = () => (
  <EditorialNarrativeSection
    id="editorial-training"
    numeral="II"
    category="Pretraining & Architecture"
    title={`Trained from Scratch on ${pretrainData.curriculum.totalDatasets} Curated Open Datasets`}
    bgVariant="cream"
    leadParagraph={`A foundation model is only as trustworthy as the corpus it absorbs and the mathematical discipline of its attention layers. Every weight in the Helios family was trained from first principles on ${pretrainData.curriculum.totalDatasets} open-access scientific, literary, and programming datasets.`}
    paragraphs={[
      `Rather than scraping uncurated web dumps filled with synthetic spam and duplicated boilerplate, the Helios pretraining pipeline ingests ${pretrainData.curriculum.totalDatasets} vetted datasets across 16 foundational domains—including formal mathematics, systems engineering (Python, Rust, TypeScript, C, and SQL), clinical biomedicine, classical philosophy, and bilingual English and Filipino/Tagalog literature. Every document undergoes MinHash deduplication, heuristic quality filtering, and cryptographic SHA-256 provenance logging.`,
      "Standard English-centric tokenizers fracture agglutinative Filipino morphology (such as mag-, nag-, -in-, and -um- affixes) and multi-space source code indentation into inefficient single-character fragments. Helios solves this with custom-trained Byte-Level BPE and Viterbi Unigram tokenizers engineered directly in-repo. By combining 256 raw byte fallbacks with priority syntactic merges, Helios achieves zero out-of-vocabulary (<unk>) failures across any UTF-8 stream while preserving high token fertility.",
      "At the heart of the model lies the Panoptes Transformer architecture. By pairing Pre-RMSNorm and SwiGLU feed-forward networks with Grouped-Query Attention (4:1 and 3:1 GQA) and Dynamic NTK High-Theta Rotary Position Embeddings (scaled up to θ = 10,000,000), Helios compresses its Key-Value cache by up to 8×. This architectural efficiency allows a 256,000-token context window—equivalent to an entire 500-page technical book or multi-module codebase—to run smoothly inside a standard 16GB RAM laptop."
    ]}
    marginalia={[
      { label: "Pretraining Corpus", value: `${pretrainData.curriculum.totalDatasets} Open Datasets · 16 Domains` },
      { label: "Tokenizer Architecture", value: "Byte-Level BPE & Viterbi Unigram" },
      { label: "Max Context Window", value: "256K Tokens (θ = 10,000,000)" },
      { label: "Attention Compression", value: "4:1 & 3:1 GQA + INT8 KV Cache" },
    ]}
    specGrid={[
      {
        heading: "Bilingual & Code Fertility",
        detail: "First-class morphological compression across English, Filipino/Tagalog, and 14 programming languages without token inflation."
      },
      {
        heading: "Multimodal ViT Integration",
        detail: "Core (1.13B) and Flagship (5.80B) integrate a 14×14 patch Vision Transformer encoder bridged via a 2-layer GELU projector."
      },
      {
        heading: "Atomic 6-State Checkpointing",
        detail: "POSIX-atomic checkpointing preserves model weights, optimizer states, RNG seeds, and exact dataloader offsets across interruptions."
      },
      {
        heading: "4-Bit QLoRA Fine-Tuning",
        detail: "Native support for NF4 double-quantized PEFT adapters across all seven attention and SwiGLU projections on a single T4 GPU."
      }
    ]}
  />
);

export const EditorialHonestySection: React.FC = () => (
  <EditorialNarrativeSection
    id="editorial-honesty"
    numeral="III"
    category="Epistemic Honesty & Sovereignty"
    title="Calibrated Uncertainty, Zero Contamination, and Local Ownership"
    bgVariant="sand"
    leadParagraph="When a closed commercial model encounters a question beyond its knowledge boundary, it frequently fabricates citations with fluent confidence. Helios is engineered around a stricter scientific ethic: verify what is known, compute what is exact, and openly state what is uncertain."
    paragraphs={[
      "To eliminate fluent hallucination, Helios pairs its autoregressive transformer with a four-tier Hierarchical Memory Architecture (HMA) and a subject-aware semantic verification gate. Before synthesizing an answer from its 134,728-node SQLite FTS5 and FAISS knowledge index, the runtime verifies entity overlap and Reciprocal Rank Fusion (RRF) confidence thresholds. If a prompt asks about an unverified event or private record outside its grounding scope, Helios explicitly abstains rather than inventing plausible fiction.",
      "Every benchmark figure published for the Helios family is produced by the in-repo HELIOS Evaluation Harness under strict zero-contamination rules. All 23 evaluation slices—spanning MMLU, GSM8K, SciQ, MedMCQA, HumanEval, and MCP Tool Orchestration—are audited against pretraining manifests to guarantee 0% test-set leakage. Evaluations run with deterministic greedy decoding (temperature = 0.0, seed = 42) and emit immutable SHA-256 verification manifests.",
      "Above all, Helios restores sovereignty to the individual developer, researcher, and institution. Released under the permissive Apache 2.0 license in SafeTensors, GGUF, and ONNX formats, Helios requires no API keys, no per-token billing, and no cloud telemetry. Whether deployed on a Raspberry Pi microcontroller at the edge or an Apple Silicon workstation in an air-gapped lab, the weights, the memory, and the intelligence remain entirely yours."
    ]}
    quote={{
      text: "An AI system worthy of scientific and civic trust must be inspectable down to its smallest tensor—and humble enough to admit when it does not know.",
      attribution: "Arron Kian Parejas, Creator of Helios"
    }}
    marginalia={[
      { label: "Contamination Audit", value: "0 / 23 Slices (100% Clean)" },
      { label: "Evaluation Protocol", value: "Greedy T=0.0 · Seed 42" },
      { label: "Grounded Retrieval", value: "134,728 Nodes (< 10ms RRF)" },
      { label: "Weights License", value: "Apache 2.0 (100% Inspectable)" },
    ]}
  />
);
