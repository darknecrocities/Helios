import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Layers, Database, Zap, ArrowRight, CheckCircle2 } from 'lucide-react';
import { KineticHeading } from '../components/KineticText';

interface MemoryTier {
  id: string;
  tier: string;
  name: string;
  latency: string;
  engine: string;
  desc: string;
  example: string;
}

const MEMORY_TIERS: MemoryTier[] = [
  {
    id: 'working',
    tier: 'TIER 01',
    name: 'Working RAM Attention Buffer',
    latency: '< 0.1 ms',
    engine: 'In-Memory Sliding Window KV Cache',
    desc: 'Maintains active conversational state with zero I/O overhead and automatic attention compaction.',
    example: '"Can you elaborate on that third point you just made?"',
  },
  {
    id: 'episodic',
    tier: 'TIER 02',
    name: 'Episodic SQLite 3 WAL',
    latency: '< 2.0 ms',
    engine: 'ACID-Compliant Relational Store',
    desc: 'Persistent cross-session memory storing interaction timelines, preferences, and verified facts.',
    example: '"Remember my preferred Python linting rules from last week."',
  },
  {
    id: 'semantic',
    tier: 'TIER 03',
    name: 'Semantic BM25 + FAISS Vector Index',
    latency: '< 10.0 ms',
    engine: 'Reciprocal Rank Fusion over 134K Chunks',
    desc: 'Sub-second hybrid dense-sparse retrieval across the pretraining corpus and external documents.',
    example: '"What does Aristotle write about daylight in De Anima?"',
  },
  {
    id: 'procedural',
    tier: 'TIER 04',
    name: 'Procedural MCP Tool Executor',
    latency: 'Deterministic',
    engine: 'JSON-RPC 2.0 Sandboxed AST Engine',
    desc: 'Safe external execution for arithmetic, code evaluation, calendar queries, and file access.',
    example: '"Calculate 4,096 * 64 and format the result as hexadecimal."',
  },
];

const CONTEXT_TIERS = [
  { label: 'NANO', tokens: '16K', percent: 12 },
  { label: 'ALPHA', tokens: '32K', percent: 25 },
  { label: 'BETA', tokens: '64K', percent: 50 },
  { label: 'CORE / PRO', tokens: '128K', percent: 75 },
  { label: 'FLAGSHIP', tokens: '256K (262,144)', percent: 100 },
];

export const CapabilitiesSection: React.FC = () => {
  const [activeMemoryId, setActiveMemoryId] = useState<string>('working');
  const activeMemory = MEMORY_TIERS.find((m) => m.id === activeMemoryId) ?? MEMORY_TIERS[0];

  return (
    <section id="capabilities" className="py-20 md:py-28 bg-[#FBF9F5] text-[#141312] border-t border-[#141312]/8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between pb-8 border-b border-[#141312]/10 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#141312]/5 border border-[#141312]/10 text-[10px] font-mono-tabular uppercase tracking-widest text-[#FF5500] mb-2">
              <span>DEVLETS TACTICAL ARCHITECTURE · PANOPTES CORE</span>
            </div>
            <KineticHeading
              text="ARCHITECTURAL CAPABILITIES"
              highlightWords={['CAPABILITIES']}
              className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#141312]"
            />
          </div>
          <p className="text-xs sm:text-sm text-[#141312]/70 max-w-md leading-relaxed">
            Engineered from first principles to deliver 256K context comprehension, 4-tier persistent memory,
            speculative decoding, and 8× KV-cache memory reduction.
          </p>
        </div>

        {/* Devlets Modular Widgets Grid */}
        <div className="mt-8 grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* WIDGET 1: Panoptes 4-Tier Hierarchical Memory Simulator (7 cols) */}
          <div className="lg:col-span-7 rounded-2xl bg-[#141312] text-[#FAF7F2] p-6 border border-white/10 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <Database className="w-4 h-4 text-[#FF5500]" aria-hidden="true" />
                  <span className="font-mono-tabular text-xs uppercase tracking-wider text-white/80 font-bold">
                    PANOPTES 4-TIER HIERARCHICAL MEMORY (HMA)
                  </span>
                </div>
                <span className="font-mono-tabular text-[10px] text-[#FF5500] bg-[#FF5500]/15 px-2 py-0.5 rounded-full">
                  LIVE INTERACTIVE
                </span>
              </div>

              {/* 4 Tier Selector Buttons */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4">
                {MEMORY_TIERS.map((m) => {
                  const isActive = m.id === activeMemoryId;
                  return (
                    <button
                      key={m.id}
                      onClick={() => setActiveMemoryId(m.id)}
                      className={`p-2.5 rounded-xl text-left border transition-all cursor-pointer ${
                        isActive
                          ? 'bg-[#FF5500]/20 border-[#FF5500] text-white'
                          : 'bg-white/5 border-white/5 text-white/60 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      <span className="font-mono-tabular text-[9px] uppercase tracking-wider text-[#FF5500] block">
                        {m.tier}
                      </span>
                      <span className="font-medium text-[11px] block mt-0.5 truncate">
                        {m.name.split(' ')[0]}
                      </span>
                      <span className="font-mono-tabular text-[9px] text-white/40 block mt-1">
                        {m.latency}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* Active Tier Interactive Detail Display */}
              <AnimatePresence mode="wait">
                <motion.div
                  key={activeMemory.id}
                  initial={{ opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -6 }}
                  transition={{ duration: 0.2 }}
                  className="mt-5 p-4 rounded-xl bg-black/40 border border-white/5"
                >
                  <div className="flex items-baseline justify-between">
                    <h3 className="font-display text-base font-bold text-white">
                      {activeMemory.name}
                    </h3>
                    <span className="font-mono-tabular text-xs font-bold text-[#FF5500]">
                      Latency: {activeMemory.latency}
                    </span>
                  </div>

                  <p className="mt-1 text-xs text-white/70 leading-relaxed">
                    {activeMemory.desc}
                  </p>

                  <div className="mt-3 pt-3 border-t border-white/5 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-[11px] font-mono-tabular">
                    <span className="text-white/40">ENGINE: {activeMemory.engine}</span>
                  </div>

                  <div className="mt-2.5 p-2 rounded-lg bg-white/5 text-[11px] font-mono-tabular text-white/90 flex items-center gap-2">
                    <ArrowRight className="w-3.5 h-3.5 text-[#FF5500] shrink-0" aria-hidden="true" />
                    <span className="truncate">{activeMemory.example}</span>
                  </div>
                </motion.div>
              </AnimatePresence>
            </div>

            <div className="mt-5 pt-3 border-t border-white/5 flex items-center justify-between text-[10px] font-mono-tabular text-white/40">
              <span>ZERO MEMORY DRIFT</span>
              <span>134,728 PRE-INDEXED KNOWLEDGE NODES</span>
            </div>
          </div>

          {/* WIDGET 2: 256K Dynamic NTK RoPE Context Engine (5 cols) */}
          <div className="lg:col-span-5 rounded-2xl bg-[#141312] text-[#FAF7F2] p-6 border border-white/10 shadow-xl flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-4 border-b border-white/10">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-[#FF5500]" aria-hidden="true" />
                  <span className="font-mono-tabular text-xs uppercase tracking-wider text-white/80 font-bold">
                    DYNAMIC NTK ROPE SCALING
                  </span>
                </div>
                <span className="font-mono-tabular text-xs text-[#FF5500] font-bold">
                  θ = 10,000,000.0
                </span>
              </div>

              <h3 className="font-display text-lg font-bold text-white mt-4">
                256K Ultra-Long Context
              </h3>
              <p className="text-xs text-white/70 mt-1 leading-relaxed">
                Scaled via dynamic Neural Tangent Kernel interpolation without retraining, preserving local token sharpness across novel-length documents.
              </p>

              {/* Context Scaling Progress Bars */}
              <div className="mt-5 space-y-2.5 font-mono-tabular">
                {CONTEXT_TIERS.map((tier) => (
                  <div key={tier.label}>
                    <div className="flex justify-between text-[10px] mb-1">
                      <span className="text-white/60">{tier.label}</span>
                      <span className="text-white font-medium">{tier.tokens}</span>
                    </div>
                    <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                      <div
                        style={{ width: `${tier.percent}%` }}
                        className={`h-full rounded-full ${
                          tier.percent === 100 ? 'bg-[#FF5500]' : 'bg-white/40'
                        }`}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-5 pt-3 border-t border-white/5 flex items-center justify-between text-[10px] font-mono-tabular text-white/40">
              <span>NIAH PASS RATE: 99.8%</span>
              <span>262,144 MAX TOKENS</span>
            </div>
          </div>

          {/* WIDGET 3: 8× Grouped-Query Attention (6 cols) */}
          <div className="lg:col-span-6 rounded-2xl bg-white p-6 border border-[#141312]/10 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-[#141312]/10">
                <span className="font-mono-tabular text-[10px] uppercase tracking-widest text-[#FF5500] font-bold">
                  MEMORY EFFICIENCY
                </span>
                <span className="font-mono-tabular text-xs font-bold text-[#141312]">8:1 GQA RATIO</span>
              </div>

              <h3 className="font-display text-lg font-bold text-[#141312] mt-3">
                8× Grouped-Query Attention (GQA)
              </h3>
              <p className="text-xs text-[#141312]/70 mt-1 leading-relaxed">
                Conventional Multi-Head Attention allocates dedicated Key-Value caches for each Query head.
                Helios clusters 8 Query heads per Key-Value pair, slashing VRAM consumption by 75% while maintaining parity.
              </p>

              <div className="mt-4 grid grid-cols-2 gap-3 p-3 rounded-xl bg-[#FBF9F5] border border-[#141312]/8 text-center font-mono-tabular text-xs">
                <div>
                  <span className="text-[10px] text-[#141312]/50 block">STANDARD MHA KV CACHE</span>
                  <span className="text-sm font-bold text-red-600">288 MB</span>
                </div>
                <div>
                  <span className="text-[10px] text-[#FF5500] block font-bold">HELIOS GQA KV CACHE</span>
                  <span className="text-sm font-bold text-[#FF5500]">36 MB (8× Savings)</span>
                </div>
              </div>
            </div>

            <div className="mt-4 flex items-center gap-1.5 text-[10px] font-mono-tabular text-[#141312]/50">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#FF5500]" aria-hidden="true" />
              <span>Runs 5.80B Flagship in 6.4GB RAM without offloading penalties</span>
            </div>
          </div>

          {/* WIDGET 4: Speculative Decoding Acceleration (6 cols) */}
          <div className="lg:col-span-6 rounded-2xl bg-white p-6 border border-[#141312]/10 shadow-xs flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-3 border-b border-[#141312]/10">
                <span className="font-mono-tabular text-[10px] uppercase tracking-widest text-[#FF5500] font-bold">
                  DUAL-ENGINE SPECULATION
                </span>
                <span className="font-mono-tabular text-xs font-bold text-[#141312]">2.8× INFERENCE SPEEDUP</span>
              </div>

              <h3 className="font-display text-lg font-bold text-[#141312] mt-3">
                Nano + Flagship Speculative Tandem
              </h3>
              <p className="text-xs text-[#141312]/70 mt-1 leading-relaxed">
                Helios Nano (17.84M) acts as an ultra-fast speculative draft engine. It generates 4 tokens in 3ms,
                which Flagship (5.80B) validates in a single parallel forward pass with mathematical zero perplexity loss.
              </p>

              <div className="mt-4 p-3 rounded-xl bg-[#FBF9F5] border border-[#141312]/8 flex items-center justify-between font-mono-tabular text-xs">
                <div className="text-left">
                  <span className="text-[10px] text-[#141312]/50 block">DRAFT γ=4 ACCEPTANCE RATE</span>
                  <span className="text-sm font-bold text-[#FF5500]">84.2%</span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-[#141312]/50 block">EFFECTIVE GENERATION</span>
                  <span className="text-sm font-bold text-[#141312]">98 tokens / sec</span>
                </div>
              </div>
            </div>

            <div className="mt-4 flex items-center gap-1.5 text-[10px] font-mono-tabular text-[#141312]/50">
              <Zap className="w-3.5 h-3.5 text-[#FF5500]" aria-hidden="true" />
              <span>Full KV-state sharing between Nano draft and Flagship verifier</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
