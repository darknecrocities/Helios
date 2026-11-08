import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BarChart3, Cpu, Eye, Layers, Sparkles, Wrench, Zap } from 'lucide-react';
import {
  HELIOS_FAMILY_METRICS,
  HELIOS_FAMILY_COMPARISON_LIST,
  type HeliosMetricKey,
} from '../data/comparisonData';
import { BespokeSeal } from '../components/BespokeSeals';
import { ScrollZoomSection } from '../components/ScrollZoomSection';

export const ComparisonSection: React.FC = () => {
  const [activeMetricKey, setActiveMetricKey] = useState<HeliosMetricKey>('capability');
  const [hoveredModelId, setHoveredModelId] = useState<string | null>('flagship-6b');

  const activeMetric =
    HELIOS_FAMILY_METRICS.find((m) => m.id === activeMetricKey) ?? HELIOS_FAMILY_METRICS[0];

  // Compute SVG polyline points across the 6 models for the animated curve overlay
  const chartPoints = HELIOS_FAMILY_COMPARISON_LIST.map((model, idx) => {
    const pct = activeMetric.getNormalizedPercent(model);
    const x = 8 + idx * 16.8; // 8% to 92%
    const y = 92 - pct * 0.76; // 92% (bottom) to 16% (top)
    return { x, y, pct, model };
  });

  const svgPathD = chartPoints
    .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x} ${pt.y}`)
    .join(' ');

  return (
    <ScrollZoomSection id="comparison" className="relative py-28 px-4 sm:px-8 overflow-hidden">
      <div className="max-w-6xl mx-auto">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
          <div>
            <p className="text-[11px] font-mono uppercase tracking-[0.24em] text-[#E85D2A] mb-2">
              Internal Family Telemetry · Nano to Flagship
            </p>
            <h2 className="font-serif text-3xl sm:text-5xl font-bold tracking-tight text-[#171412]">
              HELIOS <span className="text-[#E85D2A]">FAMILY</span> COMPARISON
            </h2>
          </div>
          <p className="text-sm text-[#4A433C] max-w-md leading-relaxed">
            Compare parameter scaling, context capacity, active RAM footprint, generation speed,
            and hardware compatibility across all six Helios models.
          </p>
        </div>

        {/* Interactive Animated Metric Graph Card */}
        <div className="rounded-3xl bg-[#171412] text-[#F5F0E6] p-6 sm:p-9 border border-[#E85D2A]/25 shadow-[0_24px_70px_rgba(23,20,18,0.18)] mb-12">
          {/* Top Metric Selector Tabs */}
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-6 border-b border-white/10">
            <div>
              <div className="flex items-center gap-2 text-xs font-mono uppercase tracking-widest text-[#FF8A50]">
                <BarChart3 className="w-4 h-4 text-[#E85D2A]" aria-hidden="true" />
                <span>Interactive 6-Tier Metric Graph</span>
              </div>
              <p className="text-xs text-[#E6DEC8]/75 mt-1">{activeMetric.subtitle}</p>
            </div>

            <div className="flex flex-wrap gap-1.5 bg-white/5 p-1.5 rounded-2xl border border-white/10">
              {HELIOS_FAMILY_METRICS.map((metric) => {
                const isActive = metric.id === activeMetricKey;
                return (
                  <button
                    key={metric.id}
                    onClick={() => setActiveMetricKey(metric.id)}
                    className={`px-3 py-1.5 rounded-xl text-[11px] font-mono uppercase tracking-wider transition-all cursor-pointer ${
                      isActive
                        ? 'bg-[#E85D2A] text-white font-bold shadow-sm'
                        : 'text-[#E6DEC8]/75 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    {metric.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Animated Graph Stage (6 Models Represented with Motion Bars + Animated SVG Curve) */}
          <div className="relative mt-8 pt-6 pb-2">
            {/* Horizontal Reference Lines */}
            <div className="absolute inset-x-0 top-6 bottom-20 flex flex-col justify-between pointer-events-none">
              {[100, 75, 50, 25, 0].map((tick) => (
                <div
                  key={tick}
                  className="w-full border-b border-dashed border-white/8 flex items-center justify-end"
                >
                  <span className="text-[9px] font-mono text-white/25 pr-1 -mt-3">
                    {tick}%
                  </span>
                </div>
              ))}
            </div>

            {/* Animated SVG Trend Line Overlay across the 6 Helios Tiers */}
            <svg
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
              className="absolute inset-x-0 top-6 h-60 w-full pointer-events-none z-10 overflow-visible"
            >
              <defs>
                <linearGradient id="heliosLineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#FF8A50" stopOpacity="0.55" />
                  <stop offset="50%" stopColor="#E85D2A" stopOpacity="0.95" />
                  <stop offset="100%" stopColor="#FFB347" stopOpacity="1" />
                </linearGradient>
              </defs>
              <AnimatePresence mode="wait">
                <motion.path
                  key={activeMetricKey}
                  d={svgPathD}
                  fill="none"
                  stroke="url(#heliosLineGrad)"
                  strokeWidth="0.9"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.85, ease: 'easeOut' }}
                />
              </AnimatePresence>
            </svg>

            {/* 6 Animated Model Columns */}
            <div className="relative z-20 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 items-end min-h-[300px]">
              {HELIOS_FAMILY_COMPARISON_LIST.map((model, index) => {
                const pct = activeMetric.getNormalizedPercent(model);
                const formattedValue = activeMetric.getFormatted(model);
                const isHovered = hoveredModelId === model.id;
                const ctxK = Math.round(model.contextTokens / 1024);

                return (
                  <div
                    key={model.id}
                    onMouseEnter={() => setHoveredModelId(model.id)}
                    onClick={() => setHoveredModelId(model.id)}
                    className={`group flex flex-col items-center p-3 rounded-2xl border transition-all cursor-pointer ${
                      isHovered
                        ? 'bg-white/8 border-[#E85D2A]/80 shadow-[0_0_30px_rgba(232,93,42,0.18)]'
                        : 'bg-white/[0.03] border-white/8 hover:border-white/25'
                    }`}
                  >
                    {/* Value Pill Above Bar */}
                    <motion.div
                      key={`${model.id}-${activeMetricKey}-val`}
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05, duration: 0.3 }}
                      className="text-center mb-3"
                    >
                      <span className="inline-block px-2 py-0.5 rounded-md bg-black/50 border border-white/10 text-[10px] font-mono text-[#FF8A50]">
                        {formattedValue}
                      </span>
                    </motion.div>

                    {/* Animated Vertical Bar Stage */}
                    <div className="w-full h-44 bg-black/40 rounded-xl p-2 flex items-end justify-center gap-1.5 overflow-hidden border border-white/6 relative">
                      {activeMetricKey === 'capability' ? (
                        /* Dual Animated Bars for Reasoning + Coding */
                        <>
                          <div className="w-1/2 h-full flex flex-col justify-end items-center">
                            <motion.div
                              initial={{ height: 0 }}
                              animate={{ height: `${model.reasoningScore}%` }}
                              transition={{
                                type: 'spring',
                                stiffness: 110,
                                damping: 18,
                                delay: index * 0.06,
                              }}
                              className="w-full rounded-t-lg bg-gradient-to-t from-[#E85D2A] to-[#FF8A50]"
                              title={`Reasoning: ${model.reasoningScore}%`}
                            />
                          </div>
                          <div className="w-1/2 h-full flex flex-col justify-end items-center">
                            <motion.div
                              initial={{ height: 0 }}
                              animate={{ height: `${model.codingScore}%` }}
                              transition={{
                                type: 'spring',
                                stiffness: 110,
                                damping: 18,
                                delay: index * 0.06 + 0.05,
                              }}
                              className="w-full rounded-t-lg bg-gradient-to-t from-[#D97706] to-[#FBBF24]"
                              title={`Coding: ${model.codingScore}%`}
                            />
                          </div>
                        </>
                      ) : (
                        /* Single Animated Metric Bar */
                        <motion.div
                          key={`${model.id}-${activeMetricKey}-bar`}
                          initial={{ height: 0 }}
                          animate={{ height: `${pct}%` }}
                          transition={{
                            type: 'spring',
                            stiffness: 110,
                            damping: 18,
                            delay: index * 0.06,
                          }}
                          className={`w-full rounded-t-lg ${
                            isHovered
                              ? 'bg-gradient-to-t from-[#E85D2A] via-[#FF7A3D] to-[#FFB347]'
                              : 'bg-gradient-to-t from-[#E85D2A]/75 to-[#FF8A50]/85'
                          }`}
                        />
                      )}
                    </div>

                    {/* Model Emblem, Tier Badge (NANO..FLAGSHIP) & Greek Name */}
                    <div className="mt-3.5 flex flex-col items-center text-center">
                      <div className="flex items-center gap-1.5 mb-1">
                        <BespokeSeal
                          tierId={model.id}
                          size={18}
                          className="text-[#E85D2A]"
                        />
                        <span className="px-2 py-0.5 rounded-full bg-[#E85D2A]/20 border border-[#E85D2A]/40 text-[9px] font-mono font-bold tracking-widest text-[#FF8A50] uppercase">
                          {model.tierBadge}
                        </span>
                      </div>
                      <span className="font-serif text-sm font-bold text-white">
                        {model.greekName}
                      </span>
                      <span className="text-[10px] font-mono text-[#E6DEC8]/60">
                        {model.tiedParams} · {ctxK}K Ctx
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Legend & Selected Model Telemetry Strip */}
            <div className="mt-6 pt-4 border-t border-white/10 flex flex-wrap items-center justify-between gap-4 text-xs font-mono text-[#E6DEC8]/75">
              <div className="flex flex-wrap items-center gap-4">
                <span className="inline-flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-xs bg-[#E85D2A]" />
                  {activeMetricKey === 'capability' ? 'Reasoning Accuracy (%)' : activeMetric.label}
                </span>
                {activeMetricKey === 'capability' && (
                  <span className="inline-flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-xs bg-[#FBBF24]" />
                    Code Synthesis & Tool Accuracy (%)
                  </span>
                )}
              </div>
              <span className="text-[11px] text-[#FF8A50]">
                Speculative Decoding: Nano (Draft γ=4) + Flagship (Target) = 2.4× Speedup
              </span>
            </div>
          </div>
        </div>

        {/* Complete 6-Tier Capabilities & Hardware Compatibility Matrix */}
        <div className="rounded-3xl bg-[#FAF7F2] border border-[#171412]/12 overflow-hidden shadow-[0_18px_50px_rgba(23,20,18,0.05)]">
          <div className="px-6 py-5 bg-[#EDE6D8]/70 border-b border-[#171412]/10 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <Layers className="w-4 h-4 text-[#E85D2A]" aria-hidden="true" />
              <h3 className="font-serif text-lg font-bold text-[#171412]">
                6-Tier Capabilities, Parameters & Hardware Compatibility Matrix
              </h3>
            </div>
            <span className="text-xs font-mono text-[#7A7067]">
              Registry: <code className="text-[#171412]">parejasarronkian/Helios</code> ·{' '}
              <code className="text-[#171412]">Darknecrocities/Helios</code>
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[#171412]/10 text-[10px] font-mono uppercase tracking-wider text-[#7A7067] bg-[#F5F0E6]/60">
                  <th className="py-3.5 px-5">Model & Tier</th>
                  <th className="py-3.5 px-4">Params & Layers</th>
                  <th className="py-3.5 px-4">Context & RoPE θ</th>
                  <th className="py-3.5 px-4">Active RAM (Q4)</th>
                  <th className="py-3.5 px-4">Capabilities (Vision / MCP / Speculative)</th>
                  <th className="py-3.5 px-4">Formats</th>
                  <th className="py-3.5 px-5">Hardware Compatibility</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#171412]/8 text-xs">
                {HELIOS_FAMILY_COMPARISON_LIST.map((model) => (
                  <tr
                    key={model.id}
                    className="hover:bg-[#EDE6D8]/45 transition-colors"
                  >
                    {/* Model & Tier Badge */}
                    <td className="py-4 px-5">
                      <div className="flex items-center gap-3">
                        <BespokeSeal
                          tierId={model.id}
                          size={26}
                          className="text-[#E85D2A] shrink-0"
                        />
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-serif font-bold text-sm text-[#171412]">
                              {model.codeName}
                            </span>
                            <span className="px-2 py-0.5 rounded-full bg-[#E85D2A]/12 text-[#E85D2A] border border-[#E85D2A]/25 text-[9px] font-mono font-bold uppercase tracking-wider">
                              {model.tierBadge}
                            </span>
                          </div>
                          <code className="text-[10px] font-mono text-[#7A7067]">
                            {model.ollamaCommand}
                          </code>
                        </div>
                      </div>
                    </td>

                    {/* Params & Layers */}
                    <td className="py-4 px-4 font-mono">
                      <span className="font-bold text-[#171412] block">{model.tiedParams}</span>
                      <span className="text-[11px] text-[#7A7067]">
                        {model.layers}L × {model.dModel}d ({model.gqaRatio})
                      </span>
                    </td>

                    {/* Context & RoPE */}
                    <td className="py-4 px-4 font-mono">
                      <span className="font-bold text-[#171412] block">
                        {Math.round(model.contextTokens / 1024)}K Tokens
                      </span>
                      <span className="text-[11px] text-[#7A7067]">θ = {model.ropeTheta}</span>
                    </td>

                    {/* Active RAM */}
                    <td className="py-4 px-4 font-mono">
                      <span className="font-bold text-[#E85D2A] block">
                        {model.activeRam2K}
                      </span>
                      <span className="text-[11px] text-[#7A7067]">
                        Weights: {model.rawQ4Weights}
                      </span>
                    </td>

                    {/* Capabilities */}
                    <td className="py-4 px-4">
                      <div className="flex flex-wrap gap-1.5">
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#171412]/6 text-[#171412] font-mono text-[10px]">
                          <Zap className="w-3 h-3 text-[#E85D2A]" aria-hidden="true" />
                          {model.speculativeRole}
                        </span>
                        {model.mcpToolCalling && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#E85D2A]/12 text-[#E85D2A] font-mono text-[10px]">
                            <Wrench className="w-3 h-3" aria-hidden="true" />
                            MCP JSON-RPC
                          </span>
                        )}
                        {model.multimodalVision && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#171412] text-[#FAF7F2] font-mono text-[10px]">
                            <Eye className="w-3 h-3 text-[#FF8A50]" aria-hidden="true" />
                            ViT Vision
                          </span>
                        )}
                        <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[#171412]/6 text-[#4A433C] font-mono text-[10px]">
                          <Sparkles className="w-3 h-3 text-[#E85D2A]" aria-hidden="true" />
                          {model.speedTokensPerSec} tok/s
                        </span>
                      </div>
                    </td>

                    {/* Formats */}
                    <td className="py-4 px-4">
                      <div className="flex flex-wrap gap-1">
                        {model.supportedFormats.map((fmt: string) => (
                          <span
                            key={fmt}
                            className="px-2 py-0.5 rounded-xs bg-[#EDE6D8] text-[#171412] font-mono text-[10px]"
                          >
                            {fmt}
                          </span>
                        ))}
                      </div>
                    </td>

                    {/* Hardware Compatibility */}
                    <td className="py-4 px-5 text-[#4A433C]">
                      <div className="flex items-start gap-1.5">
                        <Cpu className="w-3.5 h-3.5 text-[#E85D2A] mt-0.5 shrink-0" aria-hidden="true" />
                        <span>{model.targetHardware}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </ScrollZoomSection>
  );
};
