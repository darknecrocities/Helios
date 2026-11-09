import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Cpu, Layers, HardDrive, Sliders, ArrowRight, Eye, Zap, Database } from 'lucide-react';
import { HELIOS_MODELS } from '../data/modelsData';
import { TIER_QUANT_MATRIX, PRECISION_MODES } from '../data/frameworkDiagramData';
import { BespokeSeal } from './BespokeSeals';

export const ModelFrameworkDiagram: React.FC = () => {
  const [selectedTierIdx, setSelectedTierIdx] = useState<number>(5);
  const [activePrecisionId, setActivePrecisionId] = useState<string>('qlora');

  const activeTier = HELIOS_MODELS[selectedTierIdx] ?? HELIOS_MODELS[5];
  const activeQuant = TIER_QUANT_MATRIX[selectedTierIdx] ?? TIER_QUANT_MATRIX[5];
  const activePrecision =
    PRECISION_MODES.find((p) => p.id === activePrecisionId) ?? PRECISION_MODES[5];

  return (
    <div className="space-y-8">
      {/* PART 1: Interactive Tier Selector & Panoptes End-to-End Architecture Blueprint */}
      <div className="rounded-2xl bg-[#141312] text-[#FAF7F2] p-5 sm:p-7 border border-white/10 shadow-xl">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-5 border-b border-white/10">
          <div>
            <span className="font-mono-tabular text-[10px] uppercase tracking-[0.2em] text-[#FF5500] block">
              INTERACTIVE ARCHITECTURAL BLUEPRINT · NANO TO FLAGSHIP
            </span>
            <h3 className="font-display text-lg sm:text-2xl font-bold text-white mt-1">
              Panoptes Transformer, Precision & LoRA/QLoRA Pipeline
            </h3>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            {HELIOS_MODELS.map((m, idx) => {
              const isSel = idx === selectedTierIdx;
              return (
                <button
                  key={m.id}
                  onClick={() => setSelectedTierIdx(idx)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono-tabular transition-all cursor-pointer flex items-center gap-1.5 ${
                    isSel
                      ? 'bg-[#FF5500] text-white font-bold shadow-sm'
                      : 'bg-white/5 text-white/65 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <span>{m.tierBadge}</span>
                  <span className="text-[10px] opacity-80">({m.tiedParams})</span>
                </button>
              );
            })}
          </div>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={activeTier.id}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.2 }}
            className="mt-6 space-y-6"
          >
            {/* Active Tier Identity Bar */}
            <div className="p-4 rounded-xl bg-white/[0.04] border border-white/10 flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <BespokeSeal tierId={activeTier.id} size={28} className="text-[#FF5500]" />
                <div>
                  <div className="flex items-center gap-2">
                    <h4 className="font-display text-base sm:text-lg font-bold text-white">
                      {activeTier.greekName} ({activeTier.tierBadge})
                    </h4>
                    <span className="font-mono-tabular text-xs text-[#FF5500] font-semibold">
                      {activeTier.tiedParams} Tied Params
                    </span>
                  </div>
                  <p className="text-xs text-white/60 font-mono-tabular">
                    {activeTier.divineDomain} ·{' '}
                    <code className="text-[#FF5500]">{activeTier.ollamaCommand}</code>
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-3 font-mono-tabular text-xs">
                <div className="px-3 py-1.5 rounded-lg bg-black/50 border border-white/10">
                  <span className="text-white/40 text-[10px] block">DIMENSIONS</span>
                  <span className="text-white font-semibold">
                    d={activeTier.dModel} · L={activeTier.layers}
                  </span>
                </div>
                <div className="px-3 py-1.5 rounded-lg bg-black/50 border border-white/10">
                  <span className="text-white/40 text-[10px] block">GQA HEADS</span>
                  <span className="text-[#FF5500] font-semibold">
                    {activeTier.headsQ}Q : {activeTier.headsKV}KV ({activeTier.gqaRatio})
                  </span>
                </div>
                <div className="px-3 py-1.5 rounded-lg bg-black/50 border border-white/10">
                  <span className="text-white/40 text-[10px] block">CONTEXT & ROPE</span>
                  <span className="text-white font-semibold">
                    {activeTier.contextWindow.split(' ')[0]} (θ={activeTier.ropeTheta})
                  </span>
                </div>
              </div>
            </div>

            {/* 5-Stage Visual Block Diagram of the Panoptes Forward + LoRA/QLoRA Pass */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-3 relative">
              <div className="p-4 rounded-xl bg-black/50 border border-white/10 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono-tabular text-[#FF5500] mb-1.5">
                    <span>STAGE 01 · INPUT</span>
                    <Eye className="w-3.5 h-3.5" />
                  </div>
                  <h5 className="font-display text-xs font-bold text-white uppercase">
                    Tokenizer & ViT Bridge
                  </h5>
                  <p className="text-[11px] text-white/65 mt-1.5 leading-relaxed">
                    32,000-vocab Byte-Level BPE / Viterbi Unigram embedding{' '}
                    <code className="text-[#FF5500]">[V, {activeTier.dModel}]</code>.
                  </p>
                </div>
                <div className="mt-3 pt-2.5 border-t border-white/10 font-mono-tabular text-[10px] text-white/80">
                  <span className="text-white/40 block">MODALITY:</span>
                  <span className={activeTier.multimodalVision ? 'text-emerald-400' : 'text-white/70'}>
                    {activeQuant.visionEncoder}
                  </span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-black/50 border border-[#FF5500]/40 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono-tabular text-[#FF5500] mb-1.5">
                    <span>STAGE 02 · ×{activeTier.layers} LAYERS</span>
                    <Layers className="w-3.5 h-3.5" />
                  </div>
                  <h5 className="font-display text-xs font-bold text-white uppercase">
                    Pre-RMSNorm + GQA ({activeTier.gqaRatio})
                  </h5>
                  <p className="text-[11px] text-white/65 mt-1.5 leading-relaxed">
                    Dynamic NTK RoPE (<code className="text-[#FF5500]">θ={activeTier.ropeTheta}</code>)
                    with symmetric INT8 KV-Cache.
                  </p>
                </div>
                <div className="mt-3 pt-2.5 border-t border-white/10 font-mono-tabular text-[10px]">
                  <span className="text-[#FF5500] block font-semibold">LoRA / QLoRA TARGETS:</span>
                  <span className="text-white/80">q_proj, k_proj, v_proj, o_proj</span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-black/50 border border-[#FF5500]/40 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono-tabular text-[#FF5500] mb-1.5">
                    <span>STAGE 03 · FFN BLOCK</span>
                    <Cpu className="w-3.5 h-3.5" />
                  </div>
                  <h5 className="font-display text-xs font-bold text-white uppercase">
                    SwiGLU MLP (d_ffn={activeTier.dFFN})
                  </h5>
                  <p className="text-[11px] text-white/65 mt-1.5 leading-relaxed">
                    SiLU-gated projections{' '}
                    <code className="text-white/90">
                      [{activeTier.dModel} → {activeTier.dFFN} → {activeTier.dModel}]
                    </code>{' '}
                    with residual connections.
                  </p>
                </div>
                <div className="mt-3 pt-2.5 border-t border-white/10 font-mono-tabular text-[10px]">
                  <span className="text-[#FF5500] block font-semibold">LoRA / QLoRA TARGETS:</span>
                  <span className="text-white/80">gate_proj, up_proj, down_proj</span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-black/50 border border-white/10 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono-tabular text-[#FF5500] mb-1.5">
                    <span>STAGE 04 · MEMORY & COT</span>
                    <Database className="w-3.5 h-3.5" />
                  </div>
                  <h5 className="font-display text-xs font-bold text-white uppercase">
                    4-Tier HMA & &lt;|thought|&gt;
                  </h5>
                  <p className="text-[11px] text-white/65 mt-1.5 leading-relaxed">
                    Working KV, SQLite WAL, FTS5+FAISS RRF (134K nodes), and MCP JSON-RPC 2.0.
                  </p>
                </div>
                <div className="mt-3 pt-2.5 border-t border-white/10 font-mono-tabular text-[10px] text-white/80">
                  <span className="text-white/40 block">INT8 MAX KV CACHE:</span>
                  <span className="text-[#FF5500] font-semibold">{activeQuant.kvCacheInt8Max}</span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-black/50 border border-white/10 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between text-[10px] font-mono-tabular text-[#FF5500] mb-1.5">
                    <span>STAGE 05 · SERIALIZATION</span>
                    <HardDrive className="w-3.5 h-3.5" />
                  </div>
                  <h5 className="font-display text-xs font-bold text-white uppercase">
                    Multi-Format Weight Export
                  </h5>
                  <p className="text-[11px] text-white/65 mt-1.5 leading-relaxed">
                    Tied LM Head projection with Speculative role:{' '}
                    <strong className="text-white">{activeTier.speculativeRole}</strong>.
                  </p>
                </div>
                <div className="mt-3 pt-2.5 border-t border-white/10 flex flex-wrap gap-1">
                  {activeTier.supportedFormats.map((fmt) => (
                    <span
                      key={fmt}
                      className="px-1.5 py-0.5 rounded bg-white/10 text-[9px] font-mono-tabular text-white/90"
                    >
                      {fmt}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Active Tier Precision Footprint Strip (FP32 -> BF16 -> INT8 -> Q4_K_M -> LoRA -> QLoRA) */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 pt-2">
              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 font-mono-tabular">
                <span className="text-[9px] text-white/45 block uppercase">FP32 Master</span>
                <span className="text-sm font-bold text-white mt-0.5 block">
                  {activeQuant.fp32Size}
                </span>
                <span className="text-[9px] text-white/40">32-bit IEEE 754</span>
              </div>
              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 font-mono-tabular">
                <span className="text-[9px] text-white/45 block uppercase">BF16 / FP16</span>
                <span className="text-sm font-bold text-white mt-0.5 block">
                  {activeQuant.fp16Bf16Size}
                </span>
                <span className="text-[9px] text-white/40">SafeTensors Native</span>
              </div>
              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 font-mono-tabular">
                <span className="text-[9px] text-white/45 block uppercase">INT8 / Q8_0</span>
                <span className="text-sm font-bold text-white mt-0.5 block">
                  {activeQuant.int8Size}
                </span>
                <span className="text-[9px] text-white/40">Symmetric 8-bit</span>
              </div>
              <div className="p-3 rounded-xl bg-[#FF5500]/15 border border-[#FF5500]/50 font-mono-tabular">
                <span className="text-[9px] text-[#FF5500] font-bold block uppercase">
                  INT4 / Q4_K_M
                </span>
                <span className="text-sm font-bold text-[#FF5500] mt-0.5 block">
                  {activeQuant.q4KmSize}
                </span>
                <span className="text-[9px] text-white/70">Ollama Default</span>
              </div>
              <div className="p-3 rounded-xl bg-white/[0.03] border border-white/10 font-mono-tabular">
                <span className="text-[9px] text-amber-400 block uppercase">LoRA (r=16, α=32)</span>
                <span className="text-sm font-bold text-white mt-0.5 block">
                  {activeQuant.loraAdapterSize}
                </span>
                <span className="text-[9px] text-white/45">{activeQuant.loraRank16Trainable}</span>
              </div>
              <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/40 font-mono-tabular">
                <span className="text-[9px] text-emerald-400 font-bold block uppercase">
                  QLoRA 4-Bit NF4
                </span>
                <span className="text-sm font-bold text-emerald-300 mt-0.5 block">
                  {activeQuant.qloraVramMin}
                </span>
                <span className="text-[9px] text-white/60">Min SFT Training RAM</span>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* PART 2: Interactive Precision, Quantization & PEFT Adapter Inspector */}
      <div className="rounded-2xl bg-[#FBF9F5] border border-[#141312]/12 p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-[#141312]/10">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-[#FF5500]" />
            <h4 className="font-display text-sm sm:text-base font-bold text-[#141312]">
              Precision Formats & Low-Rank Adaptation Architecture (FP32 · BF16 · INT8 · INT4 · LoRA · QLoRA)
            </h4>
          </div>
          <span className="font-mono-tabular text-[10px] text-[#141312]/50 uppercase">
            Click format to inspect tensor dataflow
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 mt-4">
          {PRECISION_MODES.map((mode) => {
            const isAct = mode.id === activePrecisionId;
            return (
              <button
                key={mode.id}
                onClick={() => setActivePrecisionId(mode.id)}
                className={`p-2.5 rounded-xl text-left border transition-all cursor-pointer font-mono-tabular ${
                  isAct
                    ? 'bg-[#141312] text-white border-[#141312] shadow-xs'
                    : 'bg-white text-[#141312]/75 border-[#141312]/10 hover:border-[#FF5500]'
                }`}
              >
                <span
                  className={`text-[10px] font-bold block truncate ${
                    isAct ? 'text-[#FF5500]' : 'text-[#141312]'
                  }`}
                >
                  {mode.badge}
                </span>
                <span
                  className={`text-[9px] block mt-0.5 ${
                    isAct ? 'text-white/70' : 'text-[#141312]/50'
                  }`}
                >
                  {mode.bitsPerWeight}
                </span>
              </button>
            );
          })}
        </div>

        <div className="mt-4 p-4 rounded-xl bg-white border border-[#141312]/10 grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
          <div className="lg:col-span-7 space-y-1.5">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-display font-bold text-sm text-[#141312]">
                {activePrecision.badge}
              </span>
              <span className="font-mono-tabular text-[11px] text-[#FF5500] font-semibold">
                · {activePrecision.compression}
              </span>
            </div>
            <p className="text-xs text-[#141312]/75 leading-relaxed">{activePrecision.desc}</p>
            <div className="pt-1 flex flex-wrap items-center gap-4 text-[11px] font-mono-tabular text-[#141312]/60">
              <span>
                <strong className="text-[#141312]">Container:</strong> {activePrecision.format}
              </span>
              <span>
                <strong className="text-[#141312]">Primary Role:</strong> {activePrecision.useCase}
              </span>
            </div>
          </div>

          <div className="lg:col-span-5 p-3.5 rounded-xl bg-[#141312] text-[#FAF7F2] font-mono-tabular text-[11px] space-y-1.5">
            <div className="flex items-center justify-between text-[9px] text-[#FF5500] uppercase tracking-wider">
              <span>TENSOR COMPUTATION GRAPH</span>
              <Zap className="w-3 h-3" />
            </div>
            {activePrecision.id === 'qlora' ? (
              <>
                <div className="text-white/90">
                  h = Dequant_NF4(W_base_4bit) · x + (α/r) · B_bf16 · A_bf16 · x
                </div>
                <div className="text-[10px] text-emerald-400 flex items-center gap-1">
                  <ArrowRight className="w-3 h-3 shrink-0" />
                  <span>Base frozen in 4-bit NF4 · Gradients flow only through B, A (BF16)</span>
                </div>
              </>
            ) : activePrecision.id === 'lora' ? (
              <>
                <div className="text-white/90">
                  h = W_0_bf16 · x + (32/16) · (B_[d×16] · A_[16×d]) · x
                </div>
                <div className="text-[10px] text-amber-400 flex items-center gap-1">
                  <ArrowRight className="w-3 h-3 shrink-0" />
                  <span>Merges cleanly into base weights: W_final = W_0 + (α/r)·B·A</span>
                </div>
              </>
            ) : (
              <>
                <div className="text-white/90">
                  W_quant = round(W_fp32 / scale_block) · ({activePrecision.bitsPerWeight})
                </div>
                <div className="text-[10px] text-white/65 flex items-center gap-1">
                  <ArrowRight className="w-3 h-3 text-[#FF5500] shrink-0" />
                  <span>Zero-copy mmap loading across CPU, Apple Metal MPS, and CUDA</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* PART 3: Complete 6-Tier Precision & Adapter Comparison Matrix Table */}
      <div className="overflow-x-auto rounded-2xl border border-[#141312]/12 bg-white">
        <div className="px-4 py-3 bg-[#141312] text-[#FAF7F2] flex flex-wrap items-center justify-between gap-2">
          <span className="font-display text-xs sm:text-sm font-bold tracking-wide uppercase">
            Complete 6-Tier Precision, Quantization & LoRA/QLoRA Footprint Matrix (Nano → Flagship)
          </span>
          <span className="font-mono-tabular text-[10px] text-[#FF5500]">
            FP32 · BF16/FP16 · INT8 · INT4 (Q4_K_M) · LoRA (r=16) · QLoRA (NF4)
          </span>
        </div>
        <table className="w-full text-left border-collapse font-mono-tabular text-xs">
          <thead>
            <tr className="border-b border-[#141312]/10 bg-[#F5F0E6] text-[10px] uppercase text-[#141312]/70">
              <th className="py-3 px-3.5">Model Tier</th>
              <th className="py-3 px-3">Params</th>
              <th className="py-3 px-3">FP32 (32-bit)</th>
              <th className="py-3 px-3">BF16 / FP16</th>
              <th className="py-3 px-3">INT8 (Q8_0)</th>
              <th className="py-3 px-3 text-[#FF5500]">INT4 (Q4_K_M)</th>
              <th className="py-3 px-3">LoRA (r=16) ΔW</th>
              <th className="py-3 px-3">QLoRA Min VRAM</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#141312]/8">
            {TIER_QUANT_MATRIX.map((row, idx) => {
              const isSelected = idx === selectedTierIdx;
              return (
                <tr
                  key={row.id}
                  onClick={() => setSelectedTierIdx(idx)}
                  className={`cursor-pointer transition-colors ${
                    isSelected ? 'bg-[#FF5500]/10 font-semibold' : 'hover:bg-[#FBF9F5]'
                  }`}
                >
                  <td className="py-2.5 px-3.5 font-bold text-[#141312]">
                    {row.greekName}{' '}
                    <span className="text-[10px] text-[#FF5500]">({row.tierBadge})</span>
                  </td>
                  <td className="py-2.5 px-3 text-[#141312]/80">{row.tiedParams}</td>
                  <td className="py-2.5 px-3 text-[#141312]/70">{row.fp32Size}</td>
                  <td className="py-2.5 px-3 text-[#141312]/80">{row.fp16Bf16Size}</td>
                  <td className="py-2.5 px-3 text-[#141312]/80">{row.int8Size}</td>
                  <td className="py-2.5 px-3 font-bold text-[#FF5500]">{row.q4KmSize}</td>
                  <td className="py-2.5 px-3 text-[#141312]/80">
                    {row.loraAdapterSize}{' '}
                    <span className="text-[10px] text-[#141312]/50">({row.loraRank16Trainable})</span>
                  </td>
                  <td className="py-2.5 px-3 text-emerald-700 font-semibold">{row.qloraVramMin}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
