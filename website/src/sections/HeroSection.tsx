import React, { useRef } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import { ArrowDown } from 'lucide-react';
import { OllamaLogo, HuggingFaceLogo } from '../components/PlatformLogos';

interface HeroSectionProps {
  onOpenDocs: () => void;
}

const ORBITING_PLANETS = [
  { name: 'Aether', tier: 'NANO', params: '17.8M', size: 8, radius: 135, duration: 24, startAngle: 20 },
  { name: 'Phaethon', tier: 'ALPHA', params: '100M', size: 10, radius: 185, duration: 32, startAngle: 95 },
  { name: 'Eos', tier: 'BETA', params: '303M', size: 11, radius: 235, duration: 40, startAngle: 170 },
  { name: 'Hyperion', tier: 'CORE', params: '1.13B', size: 13, radius: 290, duration: 48, startAngle: 245 },
  { name: 'Apollo', tier: 'PRO', params: '3.32B', size: 15, radius: 350, duration: 58, startAngle: 315 },
  { name: 'Sol Invictus', tier: 'FLAGSHIP', params: '5.80B', size: 18, radius: 415, duration: 68, startAngle: 55 },
];

export const HeroSection: React.FC<HeroSectionProps> = ({ onOpenDocs }) => {
  const containerRef = useRef<HTMLElement>(null);

  // Extended 260vh sticky scroll stage so user watches the text fade out, then the sun zoom & dissolve
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ['start start', 'end start'],
  });
  const smoothScroll = useSpring(scrollYProgress, { damping: 30, stiffness: 180, mass: 0.8 });

  // Phase 1 (0 -> 0.42): Title & foreground copy expand slightly and disappear cleanly
  const titleScale = useTransform(smoothScroll, [0, 0.42], [1, 1.45]);
  const titleOpacity = useTransform(smoothScroll, [0, 0.28, 0.42], [1, 0.7, 0]);
  const copyY = useTransform(smoothScroll, [0, 0.38], ['0%', '45%']);
  const copyOpacity = useTransform(smoothScroll, [0, 0.22, 0.38], [1, 0.65, 0]);

  // Phase 2 (0 -> 0.82): Sun & 6 planetary orbits zoom deeply toward the viewer
  const sunScale = useTransform(smoothScroll, [0, 0.82], [1, 4.4]);
  const sunOpacity = useTransform(smoothScroll, [0, 0.68, 0.92], [1, 0.95, 0]);
  const orbitsScale = useTransform(smoothScroll, [0, 0.78], [1, 3.1]);
  const orbitsOpacity = useTransform(smoothScroll, [0, 0.55, 0.82], [1, 0.75, 0]);
  const bgScale = useTransform(smoothScroll, [0, 1], [1, 1.48]);
  const bgOpacity = useTransform(smoothScroll, [0, 0.75, 0.96], [0.8, 0.6, 0]);

  return (
    <section
      id="hero"
      ref={containerRef}
      className="relative h-[260vh] bg-[#141312]"
    >
      {/* Sticky Full-Viewport Stage */}
      <div className="sticky top-0 h-screen w-full overflow-hidden flex flex-col items-center justify-between pt-28 pb-12 px-4 sm:px-8">
        {/* Layer 1: Full-Bleed Celestial Ruins at Dusk_Bg.png with Scroll Zoom */}
        <motion.div
          style={{ scale: bgScale, opacity: bgOpacity }}
          className="pointer-events-none absolute inset-0 z-0"
        >
          <img
            src="/assets/Celestial Ruins at Dusk_Bg.png"
            alt="Helios Sanctuary at Dusk"
            className="w-full h-full object-cover object-center"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-[#141312]/70 via-[#141312]/35 to-[#141312]" />
        </motion.div>

        {/* Layer 2: Central Sun & 6 Orbiting Planetary Models (Zooms In & Dissolves On Scroll) */}
        <div className="pointer-events-none absolute inset-0 z-10 flex items-center justify-center overflow-hidden">
          <motion.div
            style={{ scale: sunScale, opacity: sunOpacity }}
            className="relative flex items-center justify-center"
          >
            <div className="w-[280px] h-[280px] sm:w-[420px] sm:h-[420px] rounded-full bg-radial from-[#FF5500]/65 via-[#FF7700]/25 to-transparent blur-3xl" />
            <motion.img
              src="/assets/sun.webp"
              alt="Helios Sun Core"
              animate={{ rotate: 360 }}
              transition={{ duration: 120, repeat: Infinity, ease: 'linear' }}
              className="absolute w-52 h-52 sm:w-76 sm:h-76 object-contain drop-shadow-[0_0_85px_rgba(255,85,0,0.85)]"
            />
          </motion.div>

          {/* 6 Orbiting Planetary Rings with Counter-Rotated Upright Tier Labels */}
          <motion.div
            style={{ scale: orbitsScale, opacity: orbitsOpacity }}
            className="absolute inset-0 flex items-center justify-center"
          >
            {ORBITING_PLANETS.map((planet) => (
              <motion.div
                key={planet.name}
                animate={{ rotate: [planet.startAngle, planet.startAngle + 360] }}
                transition={{
                  duration: planet.duration,
                  repeat: Infinity,
                  ease: 'linear',
                }}
                style={{
                  width: planet.radius * 2,
                  height: planet.radius * 2,
                }}
                className="absolute rounded-full border border-[#FAF7F2]/18 flex items-center justify-center"
              >
                <div
                  style={{
                    width: planet.size,
                    height: planet.size,
                    top: -planet.size / 2,
                  }}
                  className="absolute left-1/2 -translate-x-1/2 rounded-full bg-[#FF5500] shadow-[0_0_18px_#FF5500] flex items-center justify-center"
                >
                  <motion.span
                    animate={{ rotate: [-planet.startAngle, -(planet.startAngle + 360)] }}
                    transition={{
                      duration: planet.duration,
                      repeat: Infinity,
                      ease: 'linear',
                    }}
                    className="hidden md:inline-flex items-center gap-1.5 absolute left-5 whitespace-nowrap font-mono-tabular text-[10px] tracking-wider uppercase text-[#FAF7F2] bg-[#141312]/85 px-2.5 py-0.5 rounded-full border border-white/15"
                  >
                    <span className="text-[#FF5500] font-bold">{planet.tier}</span>
                    <span>· {planet.name}</span>
                    <span className="text-[#FAF7F2]/60">({planet.params})</span>
                  </motion.span>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>

        {/* Layer 3: Monumental HELIOS Title Centered Over the Sun */}
        <motion.div
          style={{ scale: titleScale, opacity: titleOpacity }}
          className="relative z-20 my-auto flex flex-col items-center text-center pointer-events-none select-none"
        >
          <h1 className="font-display font-bold text-[16vw] sm:text-[13vw] lg:text-[10.5rem] leading-none tracking-[0.18em] text-[#FAF7F2] drop-shadow-[0_10px_50px_rgba(0,0,0,0.85)]">
            HELIOS
          </h1>
        </motion.div>

        {/* Layer 4: Clean Model Representation (No Creator Name on Hero) */}
        <motion.div
          style={{ y: copyY, opacity: copyOpacity }}
          className="relative z-30 max-w-2xl w-full mx-auto text-center"
        >
          <p className="font-serif-editorial text-xl sm:text-2xl md:text-3xl text-[#FAF7F2] font-normal leading-snug drop-shadow-[0_4px_20px_rgba(0,0,0,0.9)]">
            An open-weight foundation model ecosystem engineered from first principles for{' '}
            <span className="text-[#FF5500] font-semibold">everyday hardware</span> and{' '}
            <span className="text-[#FF5500] font-semibold">verifiable truth</span>.
          </p>
          <p className="mt-2.5 text-xs sm:text-sm text-[#FAF7F2]/80 leading-relaxed max-w-xl mx-auto">
            Six sovereign tiers—Nano, Alpha, Beta, Core, Pro, and Flagship (17.84M to 5.80B
            parameters)—with 256K context scaling, 8× GQA memory savings, and 4-tier recall.
          </p>

          {/* Clean Single-Row Action Bar */}
          <div className="mt-7 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#family"
              className="px-6 py-3 rounded-full bg-[#FF5500] text-white text-xs font-semibold uppercase tracking-[0.16em] hover:bg-[#E64A00] transition-colors shadow-[0_8px_28px_rgba(255,85,0,0.4)]"
            >
              Explore Models
            </a>
            <button
              onClick={onOpenDocs}
              className="px-6 py-3 rounded-full bg-[#FAF7F2] text-[#141312] text-xs font-semibold uppercase tracking-[0.16em] hover:bg-[#FF5500] hover:text-white transition-colors cursor-pointer"
            >
              Documentation
            </button>
            <a
              href="https://ollama.com/parejasarronkian"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-3 rounded-full bg-white/10 hover:bg-white/20 text-[#FAF7F2] border border-white/15 backdrop-blur-md text-xs font-mono-tabular transition-colors"
            >
              <OllamaLogo size={16} />
              <span>parejasarronkian/Helios</span>
            </a>
            <a
              href="https://huggingface.co/Darknecrocities/Helios"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-3 rounded-full bg-white/10 hover:bg-white/20 text-[#FAF7F2] border border-white/15 backdrop-blur-md text-xs font-mono-tabular transition-colors"
            >
              <HuggingFaceLogo size={16} />
              <span>Darknecrocities/Helios</span>
            </a>
          </div>

          <div className="mt-6 inline-flex items-center gap-1.5 text-[10px] font-mono-tabular uppercase tracking-[0.22em] text-[#FAF7F2]/55">
            <span>Scroll to Zoom Into the Sun</span>
            <ArrowDown className="w-3 h-3 animate-bounce text-[#FF5500]" />
          </div>
        </motion.div>
      </div>
    </section>
  );
};
