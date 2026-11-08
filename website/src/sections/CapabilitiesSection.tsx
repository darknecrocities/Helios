import React from 'react';
import { motion } from 'framer-motion';
import {
  Cpu,
  Layers,
  Database,
  ShieldCheck,
  Globe2,
  Wrench,
  Zap,
  Eye,
  CheckCircle2,
} from 'lucide-react';
import { MorphingCard } from '../components/MorphingCard';
import { KineticHeading } from '../components/KineticText';
import { ScrollZoomSection } from '../components/ScrollZoomSection';

const CONTEXT_SCALES = [
  { tier: 'NANO', tokens: '16K', width: '18%' },
  { tier: 'ALPHA', tokens: '32K', width: '32%' },
  { tier: 'BETA', tokens: '64K', width: '48%' },
  { tier: 'CORE / PRO', tokens: '128K', width: '72%' },
  { tier: 'FLAGSHIP', tokens: '256K (262,144)', width: '100%' },
];

const HMA_TIERS = [
  {
    tier: 'TIER 01 · WORKING',
    engine: 'In-RAM Sliding Token Buffer',
    latency: '< 0.1 ms',
    desc: 'Active multi-turn attention state with automatic context window compaction.',
  },
  {
    tier: 'TIER 02 · EPISODIC',
    engine: 'SQLite 3 Write-Ahead Log (WAL)',
    latency: '< 2.0 ms',
    desc: 'ACID-compliant cross-session dialogue recall and persistent user memory.',
  },
  {
    tier: 'TIER 03 · SEMANTIC',
    engine: '15K RAM BM25 + SQLite FTS5 + FAISS',
    latency: '< 10.0 ms',
    desc: 'Reciprocal Rank Fusion over 134,728 grounded records (402 MB index).',
  },
  {
    tier: 'TIER 04 · PROCEDURAL',
    engine: 'MCP JSON-RPC 2.0 + AST Evaluator',
    latency: 'Deterministic',
    desc: 'Schema-validated tool execution, exact arithmetic, and unit conversion.',
  },
];

export const CapabilitiesSection: React.FC = () => {
  return (
    <ScrollZoomSection
      id="capabilities"
      className="py-28 md:py-36 bg-[#F5F0E6] text-[#171412] border-t border-[#171412]/10"
    >
      <div className="max-w-6xl mx-auto px-4 sm:px-8">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between pb-12 border-b border-[#171412]/10 gap-6">
          <div>
            <p className="text-[11px] font-mono uppercase tracking-[0.24em] text-[#E85D2A] mb-2">
              Asymmetrical Architecture Bento · Panoptes Engine
            </p>
            <KineticHeading
              text="CORE CAPABILITIES"
              highlightWords={['CAPABILITIES']}
              className="font-serif text-3xl sm:text-5xl font-bold tracking-tight text-[#171412]"
            />
          </div>
          <p className="text-sm text-[#4A433C] max-w-md leading-relaxed">
            Engineered from first principles to unite 256K long-context comprehension, 4-tier
            persistent recall, speculative decoding, and ultra-low memory consumption.
          </p>
        </div>

        {/* Asymmetrical 12-Column Bento Grid */}
        <div className="mt-10 grid grid-cols-1 md:grid-cols-6 lg:grid-cols-12 gap-5 auto-rows-auto">
          {/* BENTO 01: Wide Hero Dark Tile (7 cols on lg) — 256K Ultra-Long Context */}
          <MorphingCard
            dark
            className="md:col-span-6 lg:col-span-7 p-7 sm:p-9 bg-[#171412] text-[#F5F0E6] border border-[#E85D2A]/30 flex flex-col justify-between shadow-[0_20px_50px_rgba(23,20,18,0.16)]"
          >
            <div>
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#E85D2A]/20 border border-[#E85D2A]/40 text-[10px] font-mono font-bold tracking-[0.16em] text-[#FF8A50] uppercase">
                  01 · Dynamic NTK High-Theta RoPE
                </span>
                <Layers className="w-5 h-5 text-[#E85D2A]" aria-hidden="true" />
              </div>

              <div className="mt-5 flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
                <h3 className="font-serif text-2xl sm:text-3xl font-bold text-white tracking-tight">
                  256K Ultra-Long Context Window
                </h3>
                <span className="font-mono text-sm font-bold text-[#FF8A50]">
                  θ = 10,000,000.0
                </span>
              </div>

              <p className="mt-2.5 text-sm text-[#E6DEC8]/80 leading-relaxed max-w-xl">
                By scaling Rotary Position Embedding base frequency from{' '}
                <code className="font-mono text-xs text-white">100K</code> (Nano) up to{' '}
                <code className="font-mono text-xs text-white">10M</code> (Flagship), Helios reads
                entire repositories, legal treaties, and books (262,144 tokens) in a single pass
                without attention decay.
              </p>

              {/* Animated Context Progression Bars across the 6 Tiers */}
              <div className="mt-6 space-y-2.5 bg-white/[0.04] p-4 rounded-2xl border border-white/10">
                {CONTEXT_SCALES.map((item, idx) => (
                  <div key={item.tier} className="space-y-1">
                    <div className="flex items-center justify-between text-[10px] font-mono">
                      <span className="text-[#E6DEC8]/75">{item.tier}</span>
                      <span className="text-[#FF8A50] font-semibold">{item.tokens} Tokens</span>
                    </div>
                    <div className="w-full h-2 rounded-full bg-white/8 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        whileInView={{ width: item.width }}
                        viewport={{ once: true }}
                        transition={{ duration: 0.7, delay: idx * 0.08, ease: 'easeOut' }}
                        className="h-full rounded-full bg-gradient-to-r from-[#E85D2A] to-[#FFB347]"
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-white/10 flex flex-wrap items-center justify-between gap-2 text-xs font-mono text-[#E6DEC8]/70">
              <span>Formula: θ_i = 10,000,000^(-2i / d_head)</span>
              <span className="text-[#FF8A50] font-semibold">262,144 Max Sequence Tokens</span>
            </div>
          </MorphingCard>

          {/* BENTO 02: Tall Right Vertical Stack Tile (5 cols, 2 rows on lg) — 4-Tier Hierarchical Memory */}
          <MorphingCard className="md:col-span-6 lg:col-span-5 lg:row-span-2 p-7 sm:p-8 bg-[#FAF7F2] border border-[#171412]/12 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between gap-3">
                <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#E85D2A]/12 border border-[#E85D2A]/25 text-[10px] font-mono font-bold tracking-[0.16em] text-[#E85D2A] uppercase">
                  02 · Cognitive Persistence
                </span>
                <Database className="w-5 h-5 text-[#171412]" aria-hidden="true" />
              </div>

              <h3 className="font-serif text-2xl sm:text-3xl font-bold text-[#171412] mt-5 tracking-tight">
                4-Tier Hierarchical Memory (HMA)
              </h3>
              <p className="mt-2 text-sm text-[#4A433C] leading-relaxed">
                Eliminates stateless amnesia by uniting working token buffers, ACID SQLite WAL
                episodic recall, two-tier FTS5 + FAISS semantic fusion, and MCP tool execution.
              </p>

              {/* 4 Stacked Memory Architecture Layers */}
              <div className="mt-6 space-y-3">
                {HMA_TIERS.map((layer) => (
                  <div
                    key={layer.tier}
                    className="p-3.5 rounded-2xl bg-[#F5F0E6] border border-[#171412]/10 hover:border-[#E85D2A]/50 transition-colors"
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className="text-[10px] font-mono font-bold tracking-wider text-[#E85D2A] uppercase">
                        {layer.tier}
                      </span>
                      <span className="px-2 py-0.5 rounded-md bg-[#171412] text-[#FAF7F2] text-[10px] font-mono">
                        {layer.latency}
                      </span>
                    </div>
                    <p className="text-xs font-mono font-bold text-[#171412]">{layer.engine}</p>
                    <p className="text-xs text-[#4A433C] mt-1 leading-relaxed">{layer.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-[#171412]/10 flex items-center justify-between text-xs font-mono">
              <span className="text-[#7A7067]">RRF Hybrid Index</span>
              <span className="text-[#E85D2A] font-bold">134,728 Grounded Records (402 MB)</span>
            </div>
          </MorphingCard>

          {/* BENTO 03: Medium Compact Tile (4 cols on lg) — 8x Lighter KV-Cache (GQA + INT8) */}
          <MorphingCard className="md:col-span-3 lg:col-span-4 p-7 bg-[#FAF7F2] border border-[#171412]/12 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-[#E85D2A]">
                  03 · GQA + INT8 KV
                </span>
                <Cpu className="w-5 h-5 text-[#171412]" aria-hidden="true" />
              </div>

              <h3 className="font-serif text-xl font-bold text-[#171412] mt-4">
                8× Lighter Memory Footprint
              </h3>
              <p className="mt-2 text-xs sm:text-sm text-[#4A433C] leading-relaxed">
                Grouped-Query Attention (`4:1` & `3:1`) plus symmetric INT8 KV-Cache quantization
                slashes 256K cache memory from `34.36 GB` down to `8.59 GB`.
              </p>

              {/* Visual Memory Comparison */}
              <div className="mt-5 p-3.5 rounded-2xl bg-[#F5F0E6] border border-[#171412]/8 space-y-2.5 text-xs font-mono">
                <div className="flex items-center justify-between text-[#7A7067]">
                  <span>Standard MHA (FP16)</span>
                  <span className="line-through">34.36 GB</span>
                </div>
                <div className="flex items-center justify-between text-[#171412] font-bold">
                  <span>Helios GQA + INT8</span>
                  <span className="text-[#E85D2A]">8.59 GB (4.0× Less)</span>
                </div>
              </div>
            </div>

            <div className="mt-5 pt-3.5 border-t border-[#171412]/10 flex items-center justify-between text-xs font-mono">
              <span className="text-[#7A7067]">Flagship Q4 Active RAM</span>
              <span className="text-[#E85D2A] font-bold">3.54 GB</span>
            </div>
          </MorphingCard>

          {/* BENTO 04: Medium Accent Tile (3 cols on lg) — 2.4x Speculative Decoding */}
          <MorphingCard className="md:col-span-3 lg:col-span-3 p-7 bg-[#EDE6D8]/90 border border-[#171412]/12 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-[#E85D2A]">
                  04 · Speculative
                </span>
                <Zap className="w-5 h-5 text-[#E85D2A]" aria-hidden="true" />
              </div>

              <div className="mt-4">
                <span className="font-serif text-3xl font-bold text-[#171412]">2.4×</span>
                <span className="block text-xs font-mono uppercase tracking-wider text-[#E85D2A] mt-0.5">
                  Lossless Speedup
                </span>
              </div>

              <p className="mt-3 text-xs sm:text-sm text-[#4A433C] leading-relaxed">
                Pairs <strong className="text-[#171412]">Aether (NANO · 17.84M)</strong> as draft
                ($\gamma = 4$) with <strong className="text-[#171412]">Sol Invictus (5.80B)</strong>{' '}
                as parallel target verifier via Leviathan rejection sampling.
              </p>
            </div>

            <div className="mt-5 pt-3.5 border-t border-[#171412]/10 flex items-center justify-between text-xs font-mono">
              <span className="text-[#7A7067]">Flagship Speed</span>
              <span className="text-[#171412] font-bold">53.3 tok/s</span>
            </div>
          </MorphingCard>

          {/* BENTO 05: Wide Bottom-Left Tile (5 cols on lg) — Native MCP & ViT Vision */}
          <MorphingCard className="md:col-span-6 lg:col-span-5 p-7 sm:p-8 bg-[#FAF7F2] border border-[#171412]/12 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#171412] text-[#FAF7F2] text-[10px] font-mono uppercase tracking-wider">
                  05 · Multimodal ViT + MCP JSON-RPC
                </span>
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-[#E85D2A]" aria-hidden="true" />
                  <Wrench className="w-4 h-4 text-[#171412]" aria-hidden="true" />
                </div>
              </div>

              <h3 className="font-serif text-2xl font-bold text-[#171412] mt-4 tracking-tight">
                Native MCP Tool Calling & ViT Vision
              </h3>
              <p className="mt-2 text-sm text-[#4A433C] leading-relaxed">
                Core (1.13B) and Flagship (5.80B) integrate a 12-layer Vision Transformer (14×14
                patches = 256 visual tokens) alongside schema-validated{' '}
                <code className="font-mono text-xs text-[#171412]">&lt;|call:tool|&gt;</code>{' '}
                JSON-RPC execution.
              </p>

              <div className="mt-4 p-3.5 rounded-2xl bg-[#171412] text-[#FAF7F2] font-mono text-[11px] overflow-x-auto">
                <code className="text-[#FF8A50]">&lt;|call:tool|&gt;</code>
                <code>{`{"name":"verify_math","args":{"expr":"262144/4"}}`}</code>
                <code className="text-[#FF8A50]">&lt;|end_call|&gt;</code>
              </div>
            </div>

            <div className="mt-5 pt-3.5 border-t border-[#171412]/10 flex items-center justify-between text-xs font-mono">
              <span className="text-[#7A7067]">Visual Prefix</span>
              <span className="text-[#E85D2A] font-bold">256 ViT Tokens + Bidirectional Mask</span>
            </div>
          </MorphingCard>

          {/* BENTO 06: Wide Bottom-Right Tile (7 cols on lg) — Epistemic Honesty & Bilingual 177 Datasets */}
          <MorphingCard className="md:col-span-6 lg:col-span-7 p-7 sm:p-8 bg-[#FAF7F2] border border-[#171412]/12 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#E85D2A]/12 border border-[#E85D2A]/25 text-[10px] font-mono font-bold text-[#E85D2A] uppercase tracking-wider">
                  06 · Epistemic Honesty & Bilingual Grounding
                </span>
                <div className="flex items-center gap-2">
                  <ShieldCheck className="w-5 h-5 text-[#E85D2A]" aria-hidden="true" />
                  <Globe2 className="w-5 h-5 text-[#171412]" aria-hidden="true" />
                </div>
              </div>

              <h3 className="font-serif text-2xl font-bold text-[#171412] mt-4 tracking-tight">
                Subject-Aware Verification Across 177 Datasets (English & Tagalog)
              </h3>
              <p className="mt-2 text-sm text-[#4A433C] leading-relaxed">
                Before emitting an answer, Helios verifies subject-entity overlap against its 16
                foundational domains (including Philippine history, law, and bilingual Tagalog) and
                cleanly abstains rather than fabricating facts when a topic is out of scope.
              </p>

              {/* 4-Step Verification Pipeline Strip */}
              <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {[
                  { step: '01', label: 'Bilingual Intent', sub: 'EN / TL Detection' },
                  { step: '02', label: 'Subject Overlap', sub: 'Entity Guard' },
                  { step: '03', label: 'RRF Retrieval', sub: 'FTS5 + FAISS' },
                  { step: '04', label: 'Epistemic Gate', sub: 'Zero Hallucination' },
                ].map((stage) => (
                  <div
                    key={stage.step}
                    className="p-3 rounded-xl bg-[#F5F0E6] border border-[#171412]/8"
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono text-[#E85D2A] font-bold">
                      <span>STEP {stage.step}</span>
                      <CheckCircle2 className="w-3.5 h-3.5" aria-hidden="true" />
                    </div>
                    <span className="block text-xs font-serif font-bold text-[#171412] mt-1">
                      {stage.label}
                    </span>
                    <span className="block text-[10px] font-mono text-[#7A7067]">{stage.sub}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-5 pt-3.5 border-t border-[#171412]/10 flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
              <span className="text-[#7A7067]">Pretraining & Grounding Corpus</span>
              <span className="text-[#171412] font-bold">
                177 Open Datasets · 16 Domains · 99.6% Grounded Accuracy
              </span>
            </div>
          </MorphingCard>
        </div>
      </div>
    </ScrollZoomSection>
  );
};
