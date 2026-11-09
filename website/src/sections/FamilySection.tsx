import React, { useRef, useState } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import { Copy, Check, Cpu, ExternalLink } from 'lucide-react';
import { HELIOS_MODELS } from '../data/modelsData';
import { BespokeSeal } from '../components/BespokeSeals';
import { OllamaLogo, HuggingFaceLogo } from '../components/PlatformLogos';
import type { HeliosModelTier } from '../types/models';

interface FamilyKineticCardProps {
  model: HeliosModelTier;
  index: number;
  smoothScroll: any;
  copiedCmd: string | null;
  onCopy: (cmd: string) => void;
}

const FamilyKineticCard: React.FC<FamilyKineticCardProps> = ({
  model,
  index,
  smoothScroll,
  copiedCmd,
  onCopy,
}) => {
  const centerProgress = index * 0.18;
  const scale = useTransform(
    smoothScroll,
    [centerProgress - 0.16, centerProgress, centerProgress + 0.16],
    [0.93, 1.06, 0.93]
  );
  const opacity = useTransform(
    smoothScroll,
    [centerProgress - 0.22, centerProgress, centerProgress + 0.22],
    [0.6, 1, 0.6]
  );
  const shadowOpacity = useTransform(
    smoothScroll,
    [centerProgress - 0.12, centerProgress, centerProgress + 0.12],
    [0, 0.35, 0]
  );

  return (
    <motion.div
      style={{
        scale,
        opacity,
      }}
      className="w-[340px] sm:w-[380px] md:w-[420px] rounded-3xl bg-[#141312] border border-white/10 hover:border-[#FF5500]/60 p-6 sm:p-7 flex flex-col justify-between shrink-0 shadow-2xl relative group transition-colors duration-300"
    >
      {/* Dynamic Solar Glow Halo */}
      <motion.div
        style={{ opacity: shadowOpacity }}
        className="pointer-events-none absolute -inset-1 rounded-3xl bg-radial from-[#FF5500]/30 to-transparent blur-xl -z-10"
      />

      <div>
        {/* Top Header: Tier Badge & Seal */}
        <div className="flex items-center justify-between pb-3 border-b border-white/10">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-0.5 rounded-full bg-[#FF5500]/15 border border-[#FF5500]/30 text-[#FF5500] text-[10px] font-mono-tabular font-bold uppercase tracking-wider">
              {model.tierBadge}
            </span>
            <span className="font-mono-tabular text-xs text-white/50">{model.tiedParams}</span>
          </div>
          <BespokeSeal tierId={model.id} size={22} className="text-[#FF5500]" />
        </div>

        {/* Model Title & Greek Epithet */}
        <div className="mt-4">
          <h3 className="font-display text-xl sm:text-2xl font-bold tracking-tight text-white group-hover:text-[#FF5500] transition-colors">
            {model.codeName}
          </h3>
          <p className="font-serif-editorial text-xs italic text-[#FF5500]/90 mt-0.5">
            {model.greekName} · {model.epithet}
          </p>
          <p className="mt-2 text-xs text-white/70 leading-relaxed min-h-[36px]">
            {model.divineDomain}
          </p>
        </div>

        {/* Target Runtime Chip */}
        <div className="mt-4 p-2.5 rounded-xl bg-white/[0.03] border border-white/5 flex items-start gap-2 text-[11px] font-mono-tabular text-white/80">
          <Cpu className="w-3.5 h-3.5 text-[#FF5500] shrink-0 mt-0.5" />
          <div className="min-w-0">
            <span className="text-[9px] uppercase tracking-wider text-white/40 block">TARGET RUNTIME</span>
            <span className="truncate block font-sans text-xs text-white/90">{model.targetHardware}</span>
          </div>
        </div>

        {/* Micro Specs 2x2 Grid */}
        <div className="mt-4 grid grid-cols-2 gap-2 text-xs font-mono-tabular">
          <div className="p-2.5 rounded-xl bg-black/40 border border-white/5">
            <span className="text-[9px] uppercase text-white/40 block">Context</span>
            <span className="font-bold text-white text-sm">{Math.round(model.contextTokens / 1024)}K</span>
            <span className="text-[10px] text-white/50 block">θ = {model.ropeTheta.toLocaleString()}</span>
          </div>

          <div className="p-2.5 rounded-xl bg-black/40 border border-white/5">
            <span className="text-[9px] uppercase text-white/40 block">GQA Attention</span>
            <span className="font-bold text-white text-sm">{model.gqaRatio}</span>
            <span className="text-[10px] text-white/50 block">{model.layers} Layers</span>
          </div>

          <div className="p-2.5 rounded-xl bg-black/40 border border-white/5">
            <span className="text-[9px] uppercase text-white/40 block">RAM (Q4 2K)</span>
            <span className="font-bold text-[#FF5500] text-sm">{model.activeRam2K}</span>
            <span className="text-[10px] text-white/50 block">Weights: {model.rawQ4Weights}</span>
          </div>

          <div className="p-2.5 rounded-xl bg-black/40 border border-white/5">
            <span className="text-[9px] uppercase text-white/40 block">Throughput</span>
            <span className="font-bold text-white text-sm">{model.speedTokensPerSec}</span>
            <span className="text-[10px] text-white/50 block">tok/sec</span>
          </div>
        </div>
      </div>

      {/* Card Footer: One-Click Run & Hubs */}
      <div className="mt-5 pt-3.5 border-t border-white/10 space-y-2">
        <div className="flex items-center justify-between gap-2 p-2 rounded-xl bg-black/60 border border-white/5 font-mono-tabular">
          <div className="flex items-center gap-1.5 min-w-0">
            <span className="text-[#FF5500] text-xs font-bold select-none">$</span>
            <span className="text-[11px] text-white/90 truncate">{model.ollamaCommand}</span>
          </div>
          <button
            onClick={() => onCopy(model.ollamaCommand)}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#FF5500] hover:bg-[#E64A00] text-white text-[10px] transition-colors shrink-0 cursor-pointer shadow-xs active:scale-95"
          >
            {copiedCmd === model.ollamaCommand ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
            <span>{copiedCmd === model.ollamaCommand ? 'Copied' : 'Copy'}</span>
          </button>
        </div>

        <div className="flex items-center justify-between text-[11px] font-mono-tabular text-white/50 px-1 pt-0.5">
          <a
            href="https://ollama.com/parejasarronkian"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-[#FF5500] transition-colors inline-flex items-center gap-1"
          >
            <OllamaLogo size={12} />
            <span>Ollama</span>
          </a>
          <a
            href="https://huggingface.co/Darknecrocities/Helios"
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-[#FF5500] transition-colors inline-flex items-center gap-1"
          >
            <HuggingFaceLogo size={12} />
            <span>Hugging Face</span>
            <ExternalLink className="w-2.5 h-2.5 opacity-60" />
          </a>
        </div>
      </div>
    </motion.div>
  );
};

export const FamilySection: React.FC = () => {
  const containerRef = useRef<HTMLElement>(null);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const handleCopy = (cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(cmd);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  // Kinetic scroll progress across 260vh track
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });
  const smoothScroll = useSpring(scrollYProgress, { damping: 28, stiffness: 180, mass: 0.6 });

  // Calibrated horizontal translation: Card 1 (0.0) -> Card 6 (0.90) with zero trailing dead space
  const scrollX = useTransform(smoothScroll, [0, 0.92], ['0%', '-68%']);
  const progressPercent = useTransform(smoothScroll, [0, 0.92], ['0%', '100%']);

  const handleScrollToTier = (idx: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const scrollTop = window.scrollY + rect.top;
    const trackHeight = containerRef.current.offsetHeight - window.innerHeight;
    const targetScroll = scrollTop + (idx * 0.18) * trackHeight;
    window.scrollTo({ top: targetScroll, behavior: 'smooth' });
  };

  return (
    <section
      id="family"
      ref={containerRef}
      className="relative h-[260vh] bg-[#0D0C0B] text-[#FAF7F2]"
    >
      {/* Sticky Full-Viewport Stage */}
      <div className="sticky top-0 h-screen w-full overflow-hidden flex flex-col justify-between py-10 px-4 sm:px-8">
        {/* Top Header Bar with Segmented Tier Jumper Tabs */}
        <div className="w-full max-w-7xl mx-auto flex flex-col md:flex-row md:items-end justify-between pb-4 border-b border-white/10 gap-4 shrink-0">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-[10px] font-mono-tabular uppercase tracking-widest text-[#FF5500] mb-1.5">
              <span>KEEBY HORIZONTAL SCROLL TRAJECTORY · 6 TIERS</span>
            </div>
            <h2 className="font-display text-2xl sm:text-4xl font-bold tracking-tight text-white">
              THE <span className="text-[#FF5500]">HELIOS</span> MODEL FAMILY
            </h2>
          </div>

          {/* Tier Jumper Segmented Control */}
          <div className="flex items-center gap-1 overflow-x-auto scrollbar-none pb-1">
            {HELIOS_MODELS.map((model, idx) => (
              <button
                key={model.id}
                onClick={() => handleScrollToTier(idx)}
                className="px-2.5 py-1 rounded-full bg-white/5 hover:bg-[#FF5500] text-white/70 hover:text-white text-[11px] font-mono-tabular uppercase tracking-wider transition-colors cursor-pointer whitespace-nowrap"
              >
                {model.tierBadge}
              </button>
            ))}
          </div>
        </div>

        {/* Kinetic Horizontal Scrolly Carousel Track */}
        <div className="relative w-full my-auto overflow-hidden py-4">
          <motion.div
            style={{ x: scrollX }}
            className="flex items-center gap-6 sm:gap-8 will-change-transform pl-4 sm:pl-12"
          >
            {HELIOS_MODELS.map((model, idx) => (
              <FamilyKineticCard
                key={model.id}
                model={model}
                index={idx}
                smoothScroll={smoothScroll}
                copiedCmd={copiedCmd}
                onCopy={handleCopy}
              />
            ))}
          </motion.div>
        </div>

        {/* Bottom Tactile Progress Bar & Unpin Indicator */}
        <div className="w-full max-w-7xl mx-auto pt-4 border-t border-white/10 flex flex-col sm:flex-row items-center justify-between text-[11px] font-mono-tabular text-white/50 gap-2 shrink-0">
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <span>01 / NANO</span>
            <div className="w-40 h-1 bg-white/10 rounded-full overflow-hidden relative">
              <motion.div
                style={{ width: progressPercent }}
                className="absolute inset-y-0 left-0 bg-[#FF5500] rounded-full"
              />
            </div>
            <span>06 / FLAGSHIP</span>
          </div>

          <div className="flex items-center gap-3">
            <span>Scroll down to traverse tiers</span>
            <span className="text-[#FF5500]">17.84M → 5.80B Parameters</span>
          </div>
        </div>
      </div>
    </section>
  );
};
