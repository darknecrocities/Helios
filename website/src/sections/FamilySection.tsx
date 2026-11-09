import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check, ChevronLeft, ChevronRight, Cpu, ExternalLink } from 'lucide-react';
import { HELIOS_MODELS } from '../data/modelsData';
import { BespokeSeal } from '../components/BespokeSeals';
import { OllamaLogo, HuggingFaceLogo } from '../components/PlatformLogos';
import type { HeliosModelTier } from '../types/models';

export const FamilySection: React.FC = () => {
  const [activeIndex, setActiveIndex] = useState<number>(5); // Default to Flagship
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const activeModel: HeliosModelTier = HELIOS_MODELS[activeIndex] ?? HELIOS_MODELS[5];

  const handleCopy = (cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(cmd);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const handlePrev = () => {
    setActiveIndex((prev) => (prev > 0 ? prev - 1 : HELIOS_MODELS.length - 1));
  };

  const handleNext = () => {
    setActiveIndex((prev) => (prev < HELIOS_MODELS.length - 1 ? prev + 1 : 0));
  };

  return (
    <section id="family" className="relative py-20 md:py-28 bg-[#0D0C0B] text-[#FAF7F2] overflow-hidden border-t border-white/5">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Header Bar */}
        <div className="flex flex-col md:flex-row md:items-end justify-between pb-8 border-b border-white/10 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[10px] font-mono-tabular uppercase tracking-widest text-[#FF5500] mb-2">
              <span>INTERACTIVE ZOOMING CAROUSEL · 6 SOVEREIGN TIERS</span>
            </div>
            <h2 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-white">
              THE <span className="text-[#FF5500]">HELIOS</span> MODEL FAMILY
            </h2>
          </div>
          <div className="text-left md:text-right">
            <span className="font-mono-tabular text-xs text-white/50 block">SELECT OR SLIDE TO INSPECT TIERS</span>
            <span className="font-mono-tabular text-[11px] text-[#FF5500]">17.84M → 5.80B PARAMETERS</span>
          </div>
        </div>

        {/* Top Tier Segmented Switcher */}
        <div className="mt-6 flex items-center gap-1.5 overflow-x-auto scrollbar-none pb-2">
          {HELIOS_MODELS.map((model, idx) => {
            const isActive = idx === activeIndex;
            return (
              <button
                key={model.id}
                onClick={() => setActiveIndex(idx)}
                className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full text-xs font-mono-tabular uppercase tracking-wider transition-all cursor-pointer whitespace-nowrap ${
                  isActive
                    ? 'bg-[#FF5500] text-white font-bold shadow-[0_0_20px_rgba(255,85,0,0.4)] scale-105'
                    : 'bg-white/5 text-white/60 hover:text-white hover:bg-white/10'
                }`}
              >
                <span>{model.tierBadge}</span>
                <span className="text-[10px] opacity-75">({model.tiedParams})</span>
              </button>
            );
          })}
        </div>

        {/* Zooming Carousel Stage */}
        <div className="mt-8 relative">
          {/* Active Card Showcase with Spring Zoom */}
          <div className="relative flex items-center justify-center">
            {/* Desktop Navigation Chevrons */}
            <button
              onClick={handlePrev}
              className="hidden lg:flex absolute -left-5 z-30 p-3 rounded-full bg-[#141312] border border-white/15 text-white hover:bg-[#FF5500] hover:border-[#FF5500] transition-all cursor-pointer shadow-xl active:scale-95"
              aria-label="Previous model"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>

            <button
              onClick={handleNext}
              className="hidden lg:flex absolute -right-5 z-30 p-3 rounded-full bg-[#141312] border border-white/15 text-white hover:bg-[#FF5500] hover:border-[#FF5500] transition-all cursor-pointer shadow-xl active:scale-95"
              aria-label="Next model"
            >
              <ChevronRight className="w-5 h-5" />
            </button>

            {/* Main Featured Tier Card */}
            <AnimatePresence mode="wait">
              <motion.div
                key={activeModel.id}
                initial={{ opacity: 0, scale: 0.95, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: -10 }}
                transition={{ duration: 0.25, ease: 'easeOut' }}
                className="w-full max-w-2xl rounded-3xl bg-[#141312] border border-[#FF5500]/50 p-6 sm:p-8 shadow-[0_0_50px_rgba(255,85,0,0.15)] flex flex-col justify-between"
              >
                <div>
                  {/* Card Header */}
                  <div className="flex items-center justify-between pb-5 border-b border-white/10">
                    <div className="flex items-center gap-3">
                      <span className="font-mono-tabular text-xs font-bold text-[#FF5500] uppercase tracking-wider">
                        TIER 0{activeIndex + 1}
                      </span>
                      <span className="px-2.5 py-0.5 rounded-full bg-[#FF5500]/20 text-[#FF5500] font-mono-tabular text-[11px] font-bold">
                        {activeModel.tierBadge}
                      </span>
                      <span className="hidden sm:inline font-mono-tabular text-[10px] text-white/40">
                        {activeModel.versionTag}
                      </span>
                    </div>
                    <BespokeSeal tierId={activeModel.id} size={30} className="text-[#FF5500]" />
                  </div>

                  {/* Title & Divine Epithet */}
                  <div className="mt-5 flex flex-col sm:flex-row sm:items-baseline justify-between gap-2">
                    <div>
                      <h3 className="font-display text-2xl sm:text-4xl font-bold text-white tracking-tight">
                        {activeModel.codeName}
                      </h3>
                      <p className="font-serif-editorial italic text-sm text-[#FF5500] mt-0.5">
                        {activeModel.greekScript} · {activeModel.epithet}
                      </p>
                    </div>
                    <span className="font-mono-tabular text-xl sm:text-2xl font-bold text-[#FF5500]">
                      {activeModel.tiedParams}
                    </span>
                  </div>

                  <p className="mt-3 text-xs sm:text-sm text-white/75 leading-relaxed">
                    {activeModel.divineDomain}
                  </p>

                  {/* Target Hardware Chip */}
                  <div className="mt-5 p-3.5 rounded-2xl bg-white/5 border border-white/5 flex items-start gap-3">
                    <Cpu className="w-4 h-4 text-[#FF5500] shrink-0 mt-0.5" aria-hidden="true" />
                    <div className="text-xs leading-snug">
                      <span className="text-white/45 block text-[9px] uppercase tracking-wider font-mono-tabular">TARGET RUNTIME HARDWARE</span>
                      <span className="text-white font-medium">{activeModel.targetHardware}</span>
                    </div>
                  </div>

                  {/* 4-Metric Grid */}
                  <div className="mt-4 grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs font-mono-tabular">
                    <div className="p-3 rounded-xl bg-black/40 border border-white/5">
                      <span className="text-white/40 block text-[9px]">CONTEXT</span>
                      <span className="text-white font-bold">{activeModel.contextWindow}</span>
                    </div>
                    <div className="p-3 rounded-xl bg-black/40 border border-white/5">
                      <span className="text-white/40 block text-[9px]">ATTENTION</span>
                      <span className="text-white font-bold">{activeModel.gqaRatio}</span>
                    </div>
                    <div className="p-3 rounded-xl bg-black/40 border border-white/5">
                      <span className="text-white/40 block text-[9px]">RAM (2K)</span>
                      <span className="text-[#FF5500] font-bold">{activeModel.activeRam2K}</span>
                    </div>
                    <div className="p-3 rounded-xl bg-black/40 border border-white/5">
                      <span className="text-white/40 block text-[9px]">THROUGHPUT</span>
                      <span className="text-white font-bold">{activeModel.speedTokensPerSec} tok/s</span>
                    </div>
                  </div>
                </div>

                {/* Card Footer: Run Command Pill */}
                <div className="mt-6 pt-5 border-t border-white/10">
                  <div className="flex items-center justify-between gap-2 bg-black/60 rounded-xl p-2.5 border border-white/5">
                    <code className="font-mono-tabular text-xs text-white/90 truncate px-2">
                      {activeModel.ollamaCommand}
                    </code>
                    <button
                      onClick={() => handleCopy(activeModel.ollamaCommand)}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#FF5500] hover:bg-[#E64A00] text-white text-xs font-mono-tabular transition-colors cursor-pointer shrink-0 active:scale-95"
                      aria-label="Copy run command"
                    >
                      {copiedCmd === activeModel.ollamaCommand ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                      <span>{copiedCmd === activeModel.ollamaCommand ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>

                  <div className="mt-4 flex items-center justify-between text-xs font-mono-tabular text-white/50">
                    <a
                      href="https://ollama.com/parejasarronkian"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-[#FF5500] inline-flex items-center gap-1.5 transition-colors"
                    >
                      <OllamaLogo size={14} />
                      <span>Ollama Registry</span>
                    </a>
                    <a
                      href={`https://huggingface.co/${activeModel.hfRepo}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-[#FF5500] inline-flex items-center gap-1.5 transition-colors"
                    >
                      <HuggingFaceLogo size={14} />
                      <span>Hugging Face</span>
                      <ExternalLink className="w-3 h-3 opacity-60" />
                    </a>
                  </div>
                </div>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Carousel Footer Navigation Bar */}
          <div className="mt-8 flex items-center justify-between pt-4 border-t border-white/10 text-xs font-mono-tabular text-white/50">
            <div className="flex items-center gap-2">
              {HELIOS_MODELS.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => setActiveIndex(idx)}
                  className={`h-2 rounded-full transition-all cursor-pointer ${
                    activeIndex === idx ? 'w-8 bg-[#FF5500]' : 'w-2 bg-white/20 hover:bg-white/40'
                  }`}
                  aria-label={`Jump to tier ${idx + 1}`}
                />
              ))}
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={handlePrev}
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white transition-colors cursor-pointer active:scale-95 lg:hidden"
                aria-label="Previous model"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <span className="text-white/80">0{activeIndex + 1} / 06</span>
              <button
                onClick={handleNext}
                className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white transition-colors cursor-pointer active:scale-95 lg:hidden"
                aria-label="Next model"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
