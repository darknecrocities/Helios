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
    <ScrollZoomSection id="family" className="relative py-28 px-4 sm:px-8 overflow-hidden">
      <div className="max-w-6xl mx-auto">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-6">
          <div>
            <p className="text-[11px] font-mono uppercase tracking-[0.24em] text-[#E85D2A] mb-2">
              Interactive Morphing Bento · 6 Tiers (Nano → Flagship)
            </p>
            <h2 className="font-serif text-3xl sm:text-5xl font-bold tracking-tight text-[#171412]">
              THE <span className="text-[#E85D2A]">HELIOS</span> MODEL FAMILY
            </h2>
          </div>
          <p className="text-sm text-[#4A433C] max-w-md leading-relaxed">
            Click any tier in the bento grid to morph and inspect its architecture,
            capabilities, and official{' '}
            <code className="font-mono text-xs text-[#171412]">parejasarronkian/Helios</code>{' '}
            deployment commands.
          </p>
        </div>

        {/* Interactive Morphing Bento Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-stretch">
          {/* Left / Top: 6-Tier Interactive Bento Selector Tiles (5 cols on lg) */}
          <div className="lg:col-span-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-3.5 content-start">
            {HELIOS_MODELS.map((model) => {
              const isSelected = model.id === activeModel.id;
              return (
                <MorphingCard
                  key={model.id}
                  dark={isSelected}
                  onClick={() => setSelectedId(model.id)}
                  className={`p-4 rounded-2xl border transition-all duration-300 text-left flex flex-col justify-between cursor-pointer ${
                    isSelected
                      ? 'bg-[#171412] text-[#F5F0E6] border-[#E85D2A] shadow-[0_18px_40px_rgba(23,20,18,0.18)]'
                      : 'bg-[#FAF7F2]/90 text-[#171412] border-[#171412]/10 hover:border-[#E85D2A]/50 hover:bg-[#FAF7F2]'
                  }`}
                >
                  <div>
                    {/* Top Row: Tier Badge (NANO / ALPHA / BETA / CORE / PRO / FLAGSHIP) + Seal */}
                    <div className="flex items-center justify-between gap-2 mb-3">
                      <span
                        className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-mono font-bold tracking-[0.16em] uppercase ${
                          isSelected
                            ? 'bg-[#E85D2A] text-white'
                            : 'bg-[#E85D2A]/12 text-[#E85D2A] border border-[#E85D2A]/25'
                        }`}
                      >
                        {model.tierBadge}
                      </span>
                      <BespokeSeal
                        tierId={model.id}
                        size={28}
                        className={isSelected ? 'text-[#FF8A50]' : 'text-[#E85D2A]'}
                      />
                    </div>

                    {/* Greek Name & Parameter Count */}
                    <h3
                      className={`font-serif text-base font-bold tracking-tight ${
                        isSelected ? 'text-[#FAF7F2]' : 'text-[#171412]'
                      }`}
                    >
                      {model.codeName}
                    </h3>
                    <p
                      className={`text-[11px] font-mono mt-0.5 ${
                        isSelected ? 'text-[#E6DEC8]/75' : 'text-[#7A7067]'
                      }`}
                    >
                      {model.tiedParams} Params · {Math.round(model.contextTokens / 1024)}K Ctx
                    </p>
                  </div>

                  {/* Bottom Quick Pill Bar */}
                  <div className="mt-4 pt-2.5 border-t border-current/10 flex items-center justify-between text-[10px] font-mono">
                    <span className={isSelected ? 'text-[#FF8A50]' : 'text-[#4A433C]'}>
                      RAM: {model.activeRam2K}
                    </span>
                    <span className={isSelected ? 'text-[#FAF7F2]/90' : 'text-[#7A7067]'}>
                      {model.speedTokensPerSec} tok/s
                    </span>
                  </div>
                </MorphingCard>
              );
            })}
          </div>

          {/* Right / Bottom: Morphing Spotlight Bento Stage (7 cols on lg) */}
          <div className="lg:col-span-7 flex">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeModel.id}
                layoutId="helios-bento-spotlight"
                initial={{ opacity: 0, y: 12, scale: 0.985 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -10, scale: 0.985 }}
                transition={{ duration: 0.32, ease: [0.16, 1, 0.3, 1] }}
                className="w-full rounded-3xl bg-[#FAF7F2] border border-[#171412]/12 p-6 sm:p-8 lg:p-9 shadow-[0_20px_60px_rgba(23,20,18,0.07)] flex flex-col justify-between"
              >
                {/* Top Spotlight Header */}
                <div>
                  <div className="flex flex-wrap items-start justify-between gap-4 pb-6 border-b border-[#171412]/10">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-16 rounded-2xl bg-[#171412] flex items-center justify-center border border-[#E85D2A]/40 shrink-0 shadow-md">
                        <BespokeSeal
                          tierId={activeModel.id}
                          size={42}
                          className="text-[#E85D2A]"
                        />
                      </div>
                      <div>
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <span className="px-2.5 py-0.5 rounded-full bg-[#E85D2A] text-white text-[10px] font-mono font-bold tracking-[0.16em] uppercase">
                            {activeModel.tierBadge} TIER
                          </span>
                          <span className="px-2.5 py-0.5 rounded-full bg-[#171412]/8 text-[#171412] text-[10px] font-mono uppercase tracking-wider">
                            {activeModel.speculativeRole}
                          </span>
                        </div>
                        <h3 className="font-serif text-2xl sm:text-3xl font-bold text-[#171412] tracking-tight">
                          {activeModel.codeName}
                        </h3>
                        <p className="text-xs font-serif italic text-[#E85D2A]">
                          {activeModel.greekScript} — {activeModel.epithet}
                        </p>
                      </div>
                    </div>

                    {/* Interactive Bento Sub-Tabs */}
                    <div className="flex items-center bg-[#EDE6D8] p-1 rounded-full border border-[#171412]/10">
                      {(
                        [
                          { id: 'overview', label: 'Overview', icon: Sparkles },
                          { id: 'specs', label: 'Tensor Specs', icon: Layers },
                          { id: 'deploy', label: 'Run Model', icon: Terminal },
                        ] as const
                      ).map((tab) => {
                        const Icon = tab.icon;
                        const active = activeTab === tab.id;
                        return (
                          <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-[11px] font-mono uppercase tracking-wider transition-all cursor-pointer ${
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
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 my-6">
                    <div className="p-3.5 rounded-2xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[10px] font-mono uppercase tracking-widest text-[#7A7067] mb-0.5">
                        Parameters
                      </span>
                      <span className="font-serif text-xl font-bold text-[#171412]">
                        {activeModel.tiedParams}
                      </span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[10px] font-mono uppercase tracking-widest text-[#7A7067] mb-0.5">
                        Context Window
                      </span>
                      <span className="font-serif text-xl font-bold text-[#171412]">
                        {Math.round(activeModel.contextTokens / 1024)}K
                      </span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[10px] font-mono uppercase tracking-widest text-[#7A7067] mb-0.5">
                        Active RAM (Q4)
                      </span>
                      <span className="font-serif text-xl font-bold text-[#E85D2A]">
                        {activeModel.activeRam2K}
                      </span>
                    </div>
                    <div className="p-3.5 rounded-2xl bg-[#F5F0E6] border border-[#171412]/8">
                      <span className="block text-[10px] font-mono uppercase tracking-widest text-[#7A7067] mb-0.5">
                        Throughput
                      </span>
                      <span className="font-serif text-xl font-bold text-[#171412]">
                        {activeModel.speedTokensPerSec}{' '}
                        <span className="text-xs font-mono">tok/s</span>
                      </span>
                    </div>
                  </div>

                  {/* Dynamic Tab Content */}
                  {activeTab === 'overview' && (
                    <div className="space-y-5">
                      <p className="text-sm text-[#4A433C] leading-relaxed">
                        {activeModel.mythologicalParallel}
                      </p>

                      {/* Capability Badges */}
                      <div className="flex flex-wrap gap-2">
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-[#171412]/6 text-[#171412]">
                          <Zap className="w-3.5 h-3.5 text-[#E85D2A]" aria-hidden="true" />
                          Role: {activeModel.speculativeRole}
                        </span>
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-[#171412]/6 text-[#171412]">
                          <Wrench className="w-3.5 h-3.5 text-[#E85D2A]" aria-hidden="true" />
                          MCP Tools: {activeModel.mcpToolCalling ? 'Native JSON-RPC' : 'Lightweight Chat'}
                        </span>
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-[#171412]/6 text-[#171412]">
                          <Eye className="w-3.5 h-3.5 text-[#E85D2A]" aria-hidden="true" />
                          Vision (ViT):{' '}
                          {activeModel.multimodalVision ? 'Supported (256 Visual Tokens)' : 'Text-Only Tier'}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1">
                        {activeModel.specializedCapabilities.map((item: string) => (
                          <div
                            key={item}
                            className="flex items-start gap-2.5 p-3 rounded-xl bg-[#F5F0E6]/75 border border-[#171412]/6 text-xs text-[#171412] leading-relaxed"
                          >
                            <span className="w-1.5 h-1.5 rounded-full bg-[#E85D2A] mt-1.5 shrink-0" />
                            <span>{item}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {activeTab === 'specs' && (
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      <div className="p-3.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[10px] font-mono uppercase text-[#7A7067] block">
                          Exact Tied Params
                        </span>
                        <span className="text-sm font-mono font-bold text-[#171412]">
                          {activeModel.tiedParamsExact.toLocaleString()}
                        </span>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[10px] font-mono uppercase text-[#7A7067] block">
                          Depth (L) × Hidden (d_model)
                        </span>
                        <span className="text-sm font-mono font-bold text-[#171412]">
                          {activeModel.layers}L × {activeModel.dModel}d
                        </span>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[10px] font-mono uppercase text-[#7A7067] block">
                          GQA Heads (Q : KV)
                        </span>
                        <span className="text-sm font-mono font-bold text-[#171412]">
                          {activeModel.headsQ}Q : {activeModel.headsKV}KV ({activeModel.gqaRatio})
                        </span>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[10px] font-mono uppercase text-[#7A7067] block">
                          SwiGLU FFN Dimension
                        </span>
                        <span className="text-sm font-mono font-bold text-[#171412]">
                          d_ff = {activeModel.dFFN}
                        </span>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[10px] font-mono uppercase text-[#7A7067] block">
                          RoPE Scaling Theta (θ)
                        </span>
                        <span className="text-sm font-mono font-bold text-[#E85D2A]">
                          {activeModel.ropeTheta}
                        </span>
                      </div>
                      <div className="p-3.5 rounded-xl bg-[#F5F0E6] border border-[#171412]/8">
                        <span className="text-[10px] font-mono uppercase text-[#7A7067] block">
                          Raw Q4 Weights
                        </span>
                        <span className="text-sm font-mono font-bold text-[#171412]">
                          {activeModel.rawQ4Weights}
                        </span>
                      </div>
                    </div>
                  )}

                  {activeTab === 'deploy' && (
                    <div className="space-y-4">
                      <div className="p-4 rounded-2xl bg-[#171412] text-[#F5F0E6] flex items-center justify-between gap-3">
                        <div className="flex items-center gap-3 overflow-x-auto">
                          <OllamaLogo size={18} className="text-white shrink-0" />
                          <code className="text-xs font-mono text-[#FAF7F2] whitespace-nowrap">
                            {activeModel.ollamaCommand}
                          </code>
                        </div>
                        <button
                          onClick={() => handleCopy(activeModel.ollamaCommand)}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#E85D2A] text-white text-[11px] font-mono uppercase tracking-wider shrink-0 cursor-pointer"
                        >
                          {copiedCmd === activeModel.ollamaCommand ? (
                            <>
                              <Check className="w-3.5 h-3.5" aria-hidden="true" />
                              Copied
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" aria-hidden="true" />
                              Copy
                            </>
                          )}
                        </button>
                      </div>

                      <div className="p-4 rounded-2xl bg-[#F5F0E6] border border-[#171412]/12 flex items-center justify-between gap-3">
                        <div className="flex items-center gap-3 overflow-x-auto">
                          <HuggingFaceLogo size={18} className="shrink-0" />
                          <code className="text-xs font-mono text-[#171412] whitespace-nowrap">
                            huggingface-cli download {activeModel.hfRepo}
                          </code>
                        </div>
                        <a
                          href="https://huggingface.co/Darknecrocities/Helios"
                          target="_blank"
                          rel="noopener noreferrer"
                          className="px-3 py-1.5 rounded-full bg-[#171412] text-[#F5F0E6] text-[11px] font-mono uppercase tracking-wider shrink-0"
                        >
                          Open HF
                        </a>
                      </div>

                      <div className="flex flex-wrap items-center gap-2 text-xs text-[#4A433C]">
                        <span className="font-mono uppercase text-[10px] text-[#7A7067]">
                          Supported Export Formats:
                        </span>
                        {activeModel.supportedFormats.map((fmt: string) => (
                          <span
                            key={fmt}
                            className="px-2.5 py-0.5 rounded-md bg-[#EDE6D8] text-[#171412] font-mono text-[11px]"
                          >
                            {fmt}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Bottom Spotlight Footer Bar */}
                <div className="mt-6 pt-5 border-t border-[#171412]/10 flex flex-wrap items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-xs text-[#4A433C]">
                    <Cpu className="w-4 h-4 text-[#E85D2A] shrink-0" aria-hidden="true" />
                    <span>
                      Target Hardware:{' '}
                      <strong className="text-[#171412]">{activeModel.targetHardware}</strong>
                    </span>
                  </div>

                  <div className="flex items-center gap-2.5">
                    <button
                      onClick={() => handleCopy(activeModel.ollamaCommand)}
                      className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[#171412] hover:bg-[#E85D2A] text-[#F5F0E6] text-xs font-mono transition-colors cursor-pointer"
                    >
                      <OllamaLogo size={14} className="text-white" />
                      <span>{activeModel.ollamaCommand}</span>
                      {copiedCmd === activeModel.ollamaCommand ? (
                        <Check className="w-3.5 h-3.5 text-[#FF8A50]" aria-hidden="true" />
                      ) : (
                        <Copy className="w-3.5 h-3.5 text-[#F5F0E6]/60" aria-hidden="true" />
                      )}
                    </button>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
      </div>
    </ScrollZoomSection>
  );
};
