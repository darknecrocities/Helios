import React, { useRef } from 'react';
import { motion, useScroll, useTransform, useSpring } from 'framer-motion';
import { ShieldCheck, Eye, Cpu } from 'lucide-react';
import { KineticHeading } from '../components/KineticText';

const AXIOMS = [
  {
    num: '01',
    title: '100% Open Weights',
    subtitle: 'Zero Black Boxes',
    desc: 'Every layer, tokenizer rule, training checkpoint, and dataset manifest is open and inspectable. No hidden corporate safety filters or proprietary lock-in.',
    icon: Eye,
  },
  {
    num: '02',
    title: 'Epistemic Honesty',
    subtitle: 'Zero Hallucinations',
    desc: 'Engineered with calibrated uncertainty heads and 4-tier memory gating. If Helios does not possess verified factual grounding, it explicitly acknowledges limits.',
    icon: ShieldCheck,
  },
  {
    num: '03',
    title: 'Everyday Hardware',
    subtitle: '8× Memory Savings',
    desc: 'Grouped-Query Attention (GQA 8:1) and Dynamic NTK RoPE context scaling allow the 5.80B Flagship to run comfortably in 6.4GB RAM on any 16GB consumer laptop.',
    icon: Cpu,
  },
];

export const AboutSection: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);

  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start end', 'end start'],
  });

  const smoothProgress = useSpring(scrollYProgress, { damping: 28, stiffness: 220, mass: 0.7 });

  // Kinetic Parallax for Greek Statue & Solar Ring
  const statueY = useTransform(smoothProgress, [0, 1], [60, -60]);
  const statueRotate = useTransform(smoothProgress, [0, 1], [-2.5, 2.5]);
  const auraScale = useTransform(smoothProgress, [0, 0.5, 1], [0.85, 1.1, 0.9]);

  return (
    <section
      id="about"
      ref={sectionRef}
      className="relative py-20 md:py-28 bg-[#FBF9F5] text-[#141312] border-t border-[#141312]/8 overflow-hidden"
    >
      {/* Subtle Monumental Background Watermark */}
      <div className="pointer-events-none select-none absolute top-12 left-0 right-0 text-center font-display font-bold text-[13vw] leading-none text-[#141312]/[0.025] tracking-[0.2em] whitespace-nowrap">
        ALETHEIA · TRUTH
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-center">
          {/* Left Column (7 cols): Editorial Narrative */}
          <div className="lg:col-span-7 space-y-6">
            <KineticHeading
              text="BUILT FOR CLARITY, NOT BLACK BOXES."
              highlightWords={['CLARITY,']}
              className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#141312] leading-[1.08]"
            />

            <p className="font-serif-editorial italic text-lg sm:text-2xl text-[#141312]/85 leading-relaxed border-l-2 border-[#FF5500] pl-4">
              “In classical antiquity, Helios was the divine sun that illuminated reality as it truly is.
              We architected the Helios model family upon that exact ideal: radical openness, epistemic humility,
              and accessibility on everyday hardware.”
            </p>

            <div className="space-y-4 text-xs sm:text-sm text-[#141312]/75 leading-relaxed">
              <p>
                Modern artificial intelligence is increasingly centralized behind closed commercial APIs.
                Architectures remain opaque, pretraining data is concealed, and when prompted on topics beyond
                their knowledge boundary, models hallucinate confident falsehoods.
              </p>
              <p>
                Conceived and engineered from the ground up by{' '}
                <strong className="text-[#141312] font-semibold">Arron Kian Parejas</strong> in the Philippines,{' '}
                <strong className="text-[#FF5500]">Helios</strong> breaks that paradigm. Every single component—from
                its custom 48,000-token BPE vocabulary and Dynamic NTK RoPE attention to its 177-dataset pretraining corpus—is
                open-source, mathematically verifiable, and engineered to run locally without cloud tethering.
              </p>
            </div>

            {/* 3 Core Axioms Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-6 border-t border-[#141312]/10">
              {AXIOMS.map((axiom) => {
                const Icon = axiom.icon;
                return (
                  <div
                    key={axiom.num}
                    className="p-4 rounded-xl bg-white/70 border border-[#141312]/8 shadow-xs hover:border-[#FF5500]/40 transition-colors"
                  >
                    <div className="flex items-center justify-between text-[#FF5500] mb-2">
                      <span className="font-mono-tabular text-[10px] font-bold tracking-wider">
                        {axiom.num}
                      </span>
                      <Icon className="w-4 h-4" aria-hidden="true" />
                    </div>
                    <h3 className="font-display text-sm font-bold text-[#141312]">
                      {axiom.title}
                    </h3>
                    <p className="text-[11px] font-mono-tabular text-[#FF5500] mt-0.5">
                      {axiom.subtitle}
                    </p>
                    <p className="mt-2 text-xs text-[#141312]/70 leading-relaxed">
                      {axiom.desc}
                    </p>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Right Column (5 cols): Kinetic Classical Sculpture Display */}
          <div className="lg:col-span-5 relative flex items-center justify-center">
            {/* Ambient Solar Halo */}
            <motion.div
              style={{ scale: auraScale }}
              className="pointer-events-none absolute w-72 h-72 sm:w-96 sm:h-96 rounded-full bg-radial from-[#FF5500]/25 via-[#FFA040]/10 to-transparent blur-3xl"
            />

            {/* Floating Classical Sun Emblem */}
            <div className="pointer-events-none absolute -top-6 right-4 sm:right-8 w-24 h-24 sm:w-28 sm:h-28 rounded-full overflow-hidden opacity-90 mix-blend-multiply shadow-md border border-[#141312]/15">
              <img
                src="/assets/helios_sun.jpeg"
                alt="Classical Helios Solar Emblem"
                className="w-full h-full object-cover"
              />
            </div>

            {/* Kinetic Marble Sculpture with Parallax & Tilt */}
            <motion.div
              style={{ y: statueY, rotateZ: statueRotate }}
              className="relative z-10 w-full max-w-sm flex flex-col items-center"
            >
              <div className="relative group">
                <img
                  src="/assets/head_statue2.png"
                  alt="Helios Classical Marble Sculpture"
                  className="w-64 sm:w-76 md:w-84 h-auto object-contain drop-shadow-[0_24px_45px_rgba(20,19,18,0.22)] transition-transform duration-500 group-hover:scale-105"
                />
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};
