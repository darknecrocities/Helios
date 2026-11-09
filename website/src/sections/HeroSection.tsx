import React, { useRef, useState } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import { ArrowDown, Copy, Check, Terminal, ExternalLink } from 'lucide-react';
import { HuggingFaceLogo } from '../components/PlatformLogos';

interface HeroSectionProps {
  onOpenDocs: () => void;
}

const HERO_TIERS = [
  { tag: 'nano', name: 'Aether Nano', params: '17.84M', ram: '120MB', arch: '12L · 192D · 16K Ctx', radius: 110, duration: 22, startAngle: 15 },
  { tag: 'alpha', name: 'Phaethon Alpha', params: '100.04M', ram: '420MB', arch: '16L · 512D · 32K Ctx', radius: 155, duration: 28, startAngle: 90 },
  { tag: 'beta', name: 'Eos Beta', params: '303.87M', ram: '1.1GB', arch: '24L · 896D · 64K Ctx', radius: 200, duration: 36, startAngle: 165 },
  { tag: 'core', name: 'Hyperion Core', params: '1.13B', ram: '2.4GB', arch: '28L · 1536D · 128K Ctx', radius: 250, duration: 44, startAngle: 240 },
  { tag: 'pro', name: 'Apollo Pro', params: '3.32B', ram: '4.8GB', arch: '32L · 2560D · 128K Ctx', radius: 305, duration: 52, startAngle: 310 },
  { tag: 'flagship', name: 'Sol Invictus', params: '5.80B', ram: '6.4GB', arch: '36L · 3584D · 256K Ctx', radius: 365, duration: 60, startAngle: 50 },
];

export const HeroSection: React.FC<HeroSectionProps> = ({ onOpenDocs }) => {
  const containerRef = useRef<HTMLElement>(null);
  const [selectedTier, setSelectedTier] = useState<string>('flagship');
  const [copied, setCopied] = useState<boolean>(false);

  const activeTierData = HERO_TIERS.find((t) => t.tag === selectedTier) ?? HERO_TIERS[5];
  const activeCommand = `ollama run parejasarronkian/Helios:${selectedTier}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(activeCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Kinetic scroll progress across 200vh track
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });
  const smoothScroll = useSpring(scrollYProgress, { damping: 28, stiffness: 180, mass: 0.6 });

  // Solar Zoom through the sun
  const sunScale = useTransform(smoothScroll, [0, 0.85], [1, 3.2]);
  const sunOpacity = useTransform(smoothScroll, [0, 0.75, 0.95], [1, 1, 0.95]);
  const orbitsScale = useTransform(smoothScroll, [0, 0.8], [1, 2.4]);
  const orbitsOpacity = useTransform(smoothScroll, [0, 0.45, 0.75], [1, 0.7, 0]);
  const bgScale = useTransform(smoothScroll, [0, 1], [1, 1.35]);

  // Foreground copy & quick-run terminal fade gently on initial scroll
  const contentY = useTransform(smoothScroll, [0, 0.42], ['0%', '30%']);
  const contentOpacity = useTransform(smoothScroll, [0, 0.35], [1, 0]);

  // Solar Daylight Bloom: seamlessly transitions the dark celestial sanctuary
  // directly into the warm #FBF9F5 daylight marble aesthetic of the About section (zero black gap)
  const daylightBloom = useTransform(smoothScroll, [0.55, 0.88, 1], [0, 0.85, 1]);

  return (
    <section
      id="hero"
      ref={containerRef}
      className="relative h-[200vh] bg-[#0D0C0B] text-[#FAF7F2]"
    >
      {/* Sticky Fullscreen Viewport Stage */}
      <div className="sticky top-0 h-screen w-full overflow-hidden flex flex-col justify-between pt-24 pb-8 px-4 sm:px-6">
        {/* Layer 1: Atmospheric Celestial Canvas */}
        <motion.div
          style={{ scale: bgScale }}
          className="pointer-events-none absolute inset-0 z-0"
        >
          <img
            src="/assets/Celestial Ruins at Dusk_Bg.png"
            alt="Helios Celestial Sanctuary"
            className="w-full h-full object-cover object-center opacity-40 mix-blend-screen"
          />
          <div className="absolute inset-0 bg-radial from-transparent via-[#0D0C0B]/60 to-[#0D0C0B]" />
        </motion.div>

        {/* Layer 2: Interactive Kinetic Solar Core & Orbital Rings */}
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center -translate-y-8 sm:-translate-y-12 overflow-hidden">
          {/* Central Sun Corona with Solar Flares */}
          <motion.div
            style={{ scale: sunScale, opacity: sunOpacity }}
            className="relative flex items-center justify-center"
          >
            <div className="w-[260px] h-[260px] sm:w-[440px] sm:h-[440px] rounded-full bg-radial from-[#FF5500]/65 via-[#FF7700]/25 to-transparent blur-3xl" />
            <div className="absolute w-[180px] h-[180px] sm:w-[300px] sm:h-[300px] rounded-full bg-radial from-[#FFA040]/75 via-[#FF5500]/30 to-transparent blur-xl" />

            <motion.img
              src="/assets/sun.webp"
              alt="Helios Sun Core"
              animate={{ rotate: 360 }}
              transition={{ duration: 140, repeat: Infinity, ease: 'linear' }}
              className="absolute w-40 h-40 sm:w-68 sm:h-68 object-contain drop-shadow-[0_0_85px_rgba(255,85,0,0.85)]"
            />
          </motion.div>

          {/* 6 Kinetic Orbital Paths with Responsive Scaling */}
          <motion.div
            style={{ scale: orbitsScale, opacity: orbitsOpacity }}
            className="absolute inset-0 flex items-center justify-center pointer-events-none scale-[0.65] sm:scale-100 origin-center"
          >
            {HERO_TIERS.map((tier) => (
              <motion.div
                key={tier.tag}
                animate={{ rotate: [tier.startAngle, tier.startAngle + 360] }}
                transition={{
                  duration: tier.duration,
                  repeat: Infinity,
                  ease: 'linear',
                }}
                style={{
                  width: tier.radius * 2,
                  height: tier.radius * 2,
                }}
                className="absolute rounded-full border border-white/[0.08] flex items-center justify-center"
              >
                <div className="absolute left-1/2 -translate-x-1/2 -top-1.5 w-2.5 h-2.5 sm:w-3 sm:h-3 rounded-full bg-[#FF5500] shadow-[0_0_12px_#FF5500] flex items-center justify-center">
                  <motion.div
                    animate={{ rotate: [-tier.startAngle, -(tier.startAngle + 360)] }}
                    transition={{
                      duration: tier.duration,
                      repeat: Infinity,
                      ease: 'linear',
                    }}
                    className="hidden lg:inline-flex items-center gap-1.5 absolute left-4 whitespace-nowrap font-mono-tabular text-[9px] uppercase tracking-wider text-[#FAF7F2] bg-[#141312]/90 px-2 py-0.5 rounded-full border border-white/10 backdrop-blur-md"
                  >
                    <span className="text-[#FF5500] font-bold">{tier.tag}</span>
                    <span className="text-white/60">· {tier.params}</span>
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Layer 3: Daylight Bloom Overlay (Crossfades seamlessly into About Section) */}
        <motion.div
          style={{ opacity: daylightBloom }}
          className="pointer-events-none absolute inset-0 z-20 bg-gradient-to-b from-[#FFA040]/30 via-[#FBF9F5] to-[#FBF9F5]"
        />

        {/* Top Developer Status Bar */}
        <motion.div
          style={{ opacity: contentOpacity }}
          className="relative z-30 w-full max-w-4xl mx-auto flex items-center justify-between text-[10px] sm:text-[11px] font-mono-tabular tracking-wider text-white/60 px-1"
        >
          <div className="inline-flex items-center gap-2 truncate">
            <span className="w-2 h-2 rounded-full bg-[#FF5500] animate-pulse shrink-0" />
            <span className="uppercase text-white/80 font-medium truncate">OPEN-WEIGHT FOUNDATION ECOSYSTEM</span>
          </div>
          <div className="hidden sm:inline-flex items-center gap-3 text-white/50 shrink-0">
            <span>OPENCLAW & HERMES READY</span>
            <span>·</span>
            <span>17.84M → 5.80B</span>
            <span>·</span>
            <span>256K ROPE CTX</span>
          </div>
        </motion.div>

        {/* Central Hero Header & Value Proposition */}
        <motion.div
          style={{ y: contentY, opacity: contentOpacity }}
          className="relative z-30 max-w-3xl w-full mx-auto text-center my-auto flex flex-col items-center px-1"
        >
          {/* Monumental Typographic Title */}
          <h1 className="font-display font-bold text-5xl sm:text-7xl md:text-8xl lg:text-9xl leading-none tracking-[0.16em] text-[#FAF7F2] drop-shadow-[0_12px_45px_rgba(0,0,0,0.9)] select-none">
            HELIOS
          </h1>

          <div className="mt-3 px-4 sm:px-6 py-3 rounded-2xl bg-[#0D0C0B]/65 backdrop-blur-md border border-white/5 max-w-2xl mx-auto shadow-xl">
            <p className="font-serif-editorial text-base sm:text-xl md:text-2xl text-[#FAF7F2] font-light leading-snug">
              The daylight architecture for open-weight intelligence—engineered from first principles for{' '}
              <span className="text-[#FF5500] font-normal underline decoration-[#FF5500]/40 underline-offset-4">everyday hardware</span> and{' '}
              <span className="text-[#FF5500] font-normal underline decoration-[#FF5500]/40 underline-offset-4">verifiable truth</span>.
            </p>

            <p className="mt-1.5 text-[11px] sm:text-xs text-[#FAF7F2]/75 leading-relaxed">
              From 17.8M edge models on a Raspberry Pi to 5.80B flagship intelligence on a standard 16GB laptop.
              Native integration with OpenClaw and Hermes Agent. Zero telemetry. 100% inspectable weights.
            </p>
          </div>

          {/* Devlets-Inspired Tactile Terminal & Runner Widget */}
          <div className="mt-4 sm:mt-5 w-full max-w-xl mx-auto rounded-2xl bg-[#141312]/95 border border-white/10 p-3 sm:p-4 backdrop-blur-xl shadow-2xl text-left">
            {/* Terminal Header & Tier Tabs */}
            <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-white/10">
              <div className="flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-[#FF5500]" aria-hidden="true" />
                <span className="font-mono-tabular text-[9px] sm:text-[10px] uppercase tracking-widest text-white/50">
                  Quick Run (Ollama)
                </span>
              </div>

              {/* 6-Tier Segmented Switcher */}
              <div className="flex items-center gap-1 bg-white/5 p-0.5 rounded-lg border border-white/5 overflow-x-auto scrollbar-none max-w-full">
                {HERO_TIERS.map((tier) => (
                  <button
                    key={tier.tag}
                    onClick={() => setSelectedTier(tier.tag)}
                    className={`px-1.5 sm:px-2 py-0.5 rounded-md text-[9px] sm:text-[10px] font-mono-tabular uppercase tracking-wider transition-all cursor-pointer whitespace-nowrap ${
                      selectedTier === tier.tag
                        ? 'bg-[#FF5500] text-white font-bold shadow-xs'
                        : 'text-white/60 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    :{tier.tag}
                  </button>
                ))}
              </div>
            </div>

            {/* Command Display & One-Click Copy */}
            <div className="mt-2.5 flex items-center justify-between gap-2 bg-black/50 rounded-xl px-3 py-2 border border-white/5 font-mono-tabular">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className="text-[#FF5500] font-bold select-none text-xs">$</span>
                <span className="text-[11px] sm:text-xs text-white/90 truncate">{activeCommand}</span>
              </div>

              <button
                onClick={handleCopy}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-[#FF5500] hover:bg-[#E64A00] text-white text-[10px] sm:text-[11px] font-medium transition-colors shrink-0 cursor-pointer shadow-xs active:scale-95 min-h-[32px]"
                aria-label="Copy run command"
              >
                {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            {/* Micro Telemetry Bar */}
            <div className="mt-2 flex items-center justify-between text-[9px] sm:text-[10px] font-mono-tabular text-white/50 px-1">
              <span>{activeTierData.name} ({activeTierData.params})</span>
              <span>Req: {activeTierData.ram}</span>
              <span className="hidden sm:inline">{activeTierData.arch}</span>
            </div>
          </div>

          {/* Quick Action Navigation Bar */}
          <div className="mt-4 sm:mt-5 flex flex-wrap items-center justify-center gap-2.5 w-full">
            <a
              href="#family"
              className="px-5 py-2 rounded-full bg-[#FAF7F2] text-[#0D0C0B] text-[11px] sm:text-xs font-semibold uppercase tracking-[0.14em] hover:bg-[#FF5500] hover:text-white transition-all shadow-md active:scale-95"
            >
              Explore 6 Tiers
            </a>
            <button
              onClick={onOpenDocs}
              className="px-5 py-2 rounded-full bg-white/10 hover:bg-white/20 text-[#FAF7F2] border border-white/15 backdrop-blur-md text-[11px] sm:text-xs font-semibold uppercase tracking-[0.14em] transition-all cursor-pointer active:scale-95"
            >
              Architecture Specs
            </button>
            <a
              href="https://huggingface.co/Darknecrocities/Helios"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-full bg-white/5 hover:bg-white/15 text-[#FAF7F2] border border-white/10 text-[11px] sm:text-xs font-mono-tabular transition-colors"
            >
              <HuggingFaceLogo size={14} />
              <span className="hidden sm:inline">Hugging Face</span>
              <ExternalLink className="w-3 h-3 opacity-60" />
            </a>
          </div>
        </motion.div>

        {/* Bottom Scroll Cue */}
        <motion.div
          style={{ opacity: contentOpacity }}
          className="relative z-30 inline-flex items-center justify-center gap-2 text-[9px] sm:text-[10px] font-mono-tabular uppercase tracking-[0.24em] text-white/45 pb-1 text-center w-full"
        >
          <span>Scroll to explore architecture</span>
          <ArrowDown className="w-3 h-3 animate-bounce text-[#FF5500]" />
        </motion.div>
      </div>
    </section>
  );
};
