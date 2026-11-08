import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check, Cpu, Layers, Sparkles, Terminal, Eye, Wrench, Zap } from 'lucide-react';
import { HELIOS_MODELS } from '../data/modelsData';
import { BespokeSeal } from '../components/BespokeSeals';
import { OllamaLogo, HuggingFaceLogo } from '../components/PlatformLogos';
import { MorphingCard } from '../components/MorphingCard';
import { ScrollZoomSection } from '../components/ScrollZoomSection';
import type { HeliosModelTier } from '../types/models';

type DetailTab = 'overview' | 'specs' | 'deploy';

export const FamilySection: React.FC = () => {
  const [selectedId, setSelectedId] = useState<string>('flagship-6b');
  const [activeTab, setActiveTab] = useState<DetailTab>('overview');
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const activeModel: HeliosModelTier =
    HELIOS_MODELS.find((m) => m.id === selectedId) ?? HELIOS_MODELS[5];

  const handleCopy = (cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(cmd);
    setTimeout(() => setCopiedCmd(null), 1800);
  };

  return (
    <ScrollZoomSection id="family" className="relative py-16 md:py-22 px-4 sm:px-6 overflow-hidden">
      <div className="max-w-5xl mx-auto">
        {/* Compact Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-8 gap-4">
          <div>
            <p className="text-[10px] font-mono uppercase tracking-[0.22em] text-[#E85D2A] mb-1.5">
              Interactive Morphing Bento · 6 Tiers (Nano → Flagship)
            </p>
            <h2 className="font-serif text-2xl sm:text-4xl font-bold tracking-tight text-[#171412]">
              THE <span className="text-[#E85D2A]">HELIOS</span> MODEL FAMILY
            </h2>
          </div>
          <p className="text-xs text-[#4A433C] max-w-sm leading-relaxed">
            Select any tier in the bento grid to inspect its architecture, capabilities, and
            official <code className="font-mono text-[11px] text-[#171412]">parejasarronkian/Helios</code> deployment commands.
          </p>
        </div>

        {/* Compact Interactive Morphing Bento Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
          {/* Left / Top: 6-Tier Compact Bento Selector Tiles (5 cols on lg) */}
          <div className="lg:col-span-5 grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-2 gap-2.5 content-start">
            {HELIOS_MODELS.map((model) => {
              const isSelected = model.id === activeModel.id;
              return (
                <MorphingCard
                  key={model.id}
                  dark={isSelected}
                  onClick={() => setSelectedId(model.id)}
                  className={`p-3 rounded-2xl border transition-all duration-300 text-left flex flex-col justify-between cursor-pointer ${
                    isSelected
                      ? 'bg-[#171412] text-[#F5F0E6] border-[#E85D2A] shadow-[0_12px_28px_rgba(23,20,18,0.16)]'
                      : 'bg-[#FAF7F2]/90 text-[#171412] border-[#171412]/10 hover:border-[#E85D2A]/50 hover:bg-[#FAF7F2]'
                  }`}
                >
                  <div>
                    {/* Top Row: Tier Badge + Seal */}
                    <div className="flex items-center justify-between gap-1.5 mb-2">
                      <span
                        className={`inline-flex items-center px-2 py-0.5 rounded-full text-[9px] font-mono font-bold tracking-[0.14em] uppercase ${
                          isSelected
                            ? 'bg-[#E85D2A] text-white'
                            : 'bg-[#E85D2A]/12 text-[#E85D2A] border border-[#E85D2A]/25'
                        }`}
                      >
                        {model.tierBadge}
                      </span>
                      <BespokeSeal
                        tierId={model.id}
                        size={20}
                        className={isSelected ? 'text-[#FF8A50]' : 'text-[#E85D2A]'}
                      />
                    </div>

                    {/* Greek Name & Parameter Count */}
                    <h3
                      className={`font-serif text-sm font-bold tracking-tight ${
                        isSelected ? 'text-[#FAF7F2]' : 'text-[#171412]'
                      }`}
                    >
                      {model.codeName}
                    </h3>
                    <p
                      className={`text-[10px] font-mono mt-0.5 ${
                        isSelected ? 'text-[#E6DEC8]/75' : 'text-[#7A7067]'
                      }`}
                    >
                      {model.tiedParams} · {Math.round(model.contextTokens / 1024)}K Ctx
                    </p>
                  </div>

                  {/* Bottom Quick Pill Bar */}
                  <div className="mt-2.5 pt-2 border-t border-current/10 flex items-center justify-between text-[9px] font-mono">
                    <span className={isSelected ? 'text-[#FF8A50]' : 'text-[#4A433C]'}>
                      {model.activeRam2K}
                    </span>
                    <span className={isSelected ? 'text-[#FAF7F2]/90' : 'text-[#7A7067]'}>
                      {model.speedTokensPerSec} tok/s
                    </span>
                  </div>
                </MorphingCard>
              );
            })}
          </div>

          {/* Right / Bottom: Compact Morphing Spotlight Bento Stage (7 cols on lg) */}
          <div className="lg:col-span-7 flex">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeModel.id}
                layoutId="helios-bento-spotlight"
                initial={{ opacity: 0, y: 8, scale: 0.99 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -8, scale: 0.99 }}
                transition={{ duration: 0.28, ease: [0.16, 1, 0.3, 1] }}
                className="w-full rounded-2xl bg-[#FAF7F2] border border-[#171412]/12 p-5 sm:p-6 shadow-[0_14px_40px_rgba(23,20,18,0.06)] flex flex-col justify-between"
              >
                {/* Top Spotlight Header */}
                <div>
                  <div className="flex flex-wrap items-start justify-between gap-3 pb-4 border-b border-[#171412]/10">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 rounded-xl bg-[#171412] flex items-center justify-center border border-[#E85D2A]/40 shrink-0 shadow-xs">
                        <BespokeSeal
                          tierId={activeModel.id}
                          size={30}
                          className="text-[#E85D2A]"
                        />
                      </div>
                      <div>
                        <div className="flex flex-wrap items-center gap-1.5 mb-0.5">
                          <span className="px-2 py-0.5 rounded-full bg-[#E85D2A] text-white text-[9px] font-mono font-bold tracking-[0.14em] uppercase">
                            {activeModel.tierBadge} TIER
                          </span>
                          <span className="px-2 py-0.5 rounded-full bg-[#171412]/8 text-[#171412] text-[9px] font-mono uppercase tracking-wider">
                            {activeModel.speculativeRole}
                          </span>
                        </div>
                        <h3 className="font-serif text-lg sm:text-xl font-bold text-[#171412] tracking-tight">
                          {activeModel.codeName}
                        </h3>
                        <p className="text-[11px] font-serif italic text-[#E85D2A]">
                          {activeModel.greekScript} — {activeModel.epithet}
                        </p>
                      </div>
                    </div>

                    {/* Interactive Bento Sub-Tabs */}
                    <div className="flex items-center bg-[#EDE6D8] p-0.5 rounded-full border border-[#171412]/10">
                      {(
                        [
                          { id: 'overview', label: 'Overview', icon: Sparkles },
                          { id: 'specs', label: 'Specs', icon: Layers },
                          { id: 'deploy', label: 'Run', icon: Terminal },
                        ] as const
                      ).map((tab) => {
                        const Icon = tab.icon;
                        const active = activeTab === tab.id;
                        return (
                          <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-[10px] font-mono uppercase tracking-wider transition-all cursor-pointer ${
                              active
                                ? 'bg-[#171412] text-[#F5F0E6] shadow-xs'
                                : 'text-[#4A433C] hover:text-[#171412]'
                            }`}
                          >
                            <Icon className="w-3 h-3 text-[#E85D2A]" aria-hidden="true" />
                            {tab.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  {/* 4 Key Metric Bento Strip */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 my-4">
                    <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[9px] font-mono uppercase tracking-widest text-[#7A7067]">
                        Parameters
                      </span>
                      <span className="font-serif text-base font-bold text-[#171412]">
                        {activeModel.tiedParams}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[9px] font-mono uppercase tracking-widest text-[#7A7067]">
                        Context
                      </span>
                      <span className="font-serif text-base font-bold text-[#171412]">
                        {Math.round(activeModel.contextTokens / 1024)}K
                      </span>
                    </div>
                    <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[9px] font-mono uppercase tracking-widest text-[#7A7067]">
                        Active RAM
                      </span>
                      <span className="font-serif text-base font-bold text-[#E85D2A]">
                        {activeModel.activeRam2K}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[9px] font-mono uppercase tracking-widest text-[#7A7067]">
                        Throughput
                      </span>
                      <span className="font-serif text-base font-bold text-[#171412]">
                        {activeModel.speedTokensPerSec}{' '}
                        <span className="text-[10px] font-mono">tok/s</span>
                      </span>
                    </div>
                  </div>

                  {/* Dynamic Tab Content */}
                  {activeTab === 'overview' && (
                    <div className="space-y-3.5">
                      <p className="text-xs text-[#4A433C] leading-relaxed">
                        {activeModel.mythologicalParallel}
                      </p>

                      {/* Capability Badges */}
                      <div className="flex flex-wrap gap-1.5">
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#171412]/6 text-[#171412]">
                          <Zap className="w-3 h-3 text-[#E85D2A]" aria-hidden="true" />
                          {activeModel.speculativeRole}
                        </span>
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#171412]/6 text-[#171412]">
                          <Wrench className="w-3 h-3 text-[#E85D2A]" aria-hidden="true" />
                          MCP: {activeModel.mcpToolCalling ? 'JSON-RPC' : 'Lightweight'}
                        </span>
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-mono bg-[#171412]/6 text-[#171412]">
                          <Eye className="w-3 h-3 text-[#E85D2A]" aria-hidden="true" />
                          Vision: {activeModel.multimodalVision ? 'ViT 256 Tokens' : 'Text-Only'}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {activeModel.specializedCapabilities.map((item: string) => (
                          <div
                            key={item}
                            className="flex items-start gap-2 p-2.5 rounded-xl bg-[#F5F0E6]/75 border border-[#171412]/6 text-[11px] text-[#171412] leading-snug"
                          >
                            <span className="w-1.5 h-1.5 rounded-full bg-[#E85D2A] mt-1 shrink-0" />
                            <span>{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {activeTab === 'specs' && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                      <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[9px] font-mono uppercase text-[#7A7067] block">
                          Exact Tied Params
                        </span>
                        <span className="text-xs font-mono font-bold text-[#171412]">
                          {activeModel.tiedParamsExact.toLocaleString()}
                        </span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[9px] font-mono uppercase text-[#7A7067] block">
                          Depth × Hidden
                        </span>
                        <span className="text-xs font-mono font-bold text-[#171412]">
                          {activeModel.layers}L × {activeModel.dModel}d
                        </span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[9px] font-mono uppercase text-[#7A7067] block">
                          GQA Heads
                        </span>
                        <span className="text-xs font-mono font-bold text-[#171412]">
                          {activeModel.headsQ}Q : {activeModel.headsKV}KV ({activeModel.gqaRatio})
                        </span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[9px] font-mono uppercase text-[#7A7067] block">
                          SwiGLU FFN
                        </span>
                        <span className="text-xs font-mono font-bold text-[#171412]">
                          d_ff = {activeModel.dFFN}
                        </span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[9px] font-mono uppercase text-[#7A7067] block">
                          RoPE Theta (θ)
                        </span>
                        <span className="text-xs font-mono font-bold text-[#E85D2A]">
                          {activeModel.ropeTheta}
                        </span>
                      </div>
                      <div className="p-2.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[9px] font-mono uppercase text-[#7A7067] block">
                          Raw Q4 Weights
                        </span>
                        <span className="text-xs font-mono font-bold text-[#171412]">
                          {activeModel.rawQ4Weights}
                        </span>
                      </div>
                    </div>
                  )}

                  {activeTab === 'deploy' && (
                    <div className="space-y-3">
                      <div className="p-3 rounded-xl bg-[#171412] text-[#F5F0E6] flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2.5 overflow-x-auto">
                          <OllamaLogo size={15} className="text-white shrink-0" />
                          <code className="text-[11px] font-mono text-[#FAF7F2] whitespace-nowrap">
                            {activeModel.ollamaCommand}
                          </code>
                        </div>
                        <button
                          onClick={() => handleCopy(activeModel.ollamaCommand)}
                          className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full bg-[#E85D2A] text-white text-[10px] font-mono uppercase tracking-wider shrink-0 cursor-pointer"
                        >
                          {copiedCmd === activeModel.ollamaCommand ? (
                            <>
                              <Check className="w-3 h-3" aria-hidden="true" />
                              Copied
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" aria-hidden="true" />
                              Copy
                            </>
                          )}
                        </button>
                      </div>

                      <div className="p-3 rounded-xl bg-[#F5F0E6] border border-[#171412]/12 flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2.5 overflow-x-auto">
                          <HuggingFaceLogo size={15} className="shrink-0" />
                          <code className="text-[11px] font-mono text-[#171412] whitespace-nowrap">
                            huggingface-cli download {activeModel.hfRepo}
                          </code>
                        </div>
                        <a
                          href="https://huggingface.co/Darknecrocities/Helios"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-2.5 py-1 rounded-full bg-[#171412] text-[#F5F0E6] text-[10px] font-mono uppercase tracking-wider shrink-0"
                        >
                          Open HF
                        </a>
                      </div>

                      <div className="flex flex-wrap items-center gap-1.5 text-xs text-[#4A433C]">
                        <span className="font-mono uppercase text-[9px] text-[#7A7067]">
                          Formats:
                        </span>
                        {activeModel.supportedFormats.map((fmt: string) => (
                          <span
                            key={fmt}
                            className="px-2 py-0.5 rounded-md bg-[#EDE6D8] text-[#171412] font-mono text-[10px]"
                          >
                            {fmt}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Bottom Spotlight Footer Bar */}
                <div className="mt-4 pt-3.5 border-t border-[#171412]/10 flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-1.5 text-[11px] text-[#4A433C]">
                    <Cpu className="w-3.5 h-3.5 text-[#E85D2A] shrink-0" aria-hidden="true" />
                    <span>
                      Hardware: <strong className="text-[#171412]">{activeModel.targetHardware}</strong>
                    </span>
                  </div>

                  <button
                    onClick={() => handleCopy(activeModel.ollamaCommand)}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#171412] hover:bg-[#E85D2A] text-[#F5F0E6] text-[11px] font-mono transition-colors cursor-pointer"
                  >
                    <OllamaLogo size={12} className="text-white" />
                    <span>{activeModel.ollamaCommand}</span>
                    {copiedCmd === activeModel.ollamaCommand ? (
                      <Check className="w-3 h-3 text-[#FF8A50]" aria-hidden="true" />
                    ) : (
                      <Copy className="w-3 h-3 text-[#F5F0E6]/60" aria-hidden="true" />
                    )}
                  </button>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </ScrollZoomSection>
  );
};
