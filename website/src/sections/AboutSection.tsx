import React from 'react';
import { KineticHeading } from '../components/KineticText';
import { ScrollZoomSection } from '../components/ScrollZoomSection';

export const AboutSection: React.FC = () => {
  return (
    <ScrollZoomSection
      id="about"
      className="py-28 md:py-40 bg-[#F5F0E6] text-[#141312] border-t border-[#141312]/10"
    >
      {/* Subtle Monumental Background Lettering */}
      <div className="pointer-events-none select-none absolute top-20 left-0 right-0 text-center font-display font-bold text-[15vw] leading-none text-[#141312]/[0.03] tracking-[0.18em] whitespace-nowrap">
        ALETHEIA · TRUTH
      </div>

      <div className="max-w-[1320px] mx-auto px-4 sm:px-8 md:px-12 relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-16 items-center">
          {/* Left 7 Columns: Clean Editorial Story */}
          <div className="lg:col-span-7 space-y-8">
            <KineticHeading
              text="BUILT FOR CLARITY, NOT BLACK BOXES."
              highlightWords={['CLARITY,']}
              className="font-display text-3xl sm:text-5xl lg:text-6xl font-bold tracking-tight text-[#141312] leading-[1.08]"
            />

            <p className="font-serif-editorial italic text-2xl sm:text-3xl text-[#141312]/85 leading-relaxed border-l-2 border-[#FF5500] pl-6">
              “In ancient Greek thought, Helios was the sun that revealed the world as it truly is.
              We built the Helios language model with that same principle: open architecture, honest
              answers, and accessible to everyone.”
            </p>

            <div className="space-y-5 text-base sm:text-lg text-[#141312]/75 leading-relaxed">
              <p>
                Most modern AI models are locked behind closed corporate servers. You cannot inspect
                how they were trained, they require expensive hardware to run, and when asked about
                something they do not know, they often guess or make up false facts.
              </p>
              <p>
                Conceived and engineered from the ground up by{' '}
                <strong className="text-[#141312] font-semibold">Arron Kian Parejas</strong> in the
                Philippines, <strong className="text-[#FF5500]">Helios</strong> is 100% open-weight
                and transparent. Every component—from its custom tokenizer and memory-saving
                attention layers to its 177-dataset pretraining corpus—is openly documented and built
                to run smoothly on everyday consumer laptops, phones, and even a Raspberry Pi.
              </p>
            </div>

            {/* Minimalist 3-Column Principles */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-8 pt-8 border-t border-[#141312]/12">
              <div>
                <span className="font-mono-tabular text-xs text-[#FF5500] uppercase tracking-widest">
                  01 / Open Weights
                </span>
                <h3 className="font-display text-lg font-bold text-[#141312] mt-1">
                  Full Transparency
                </h3>
                <p className="mt-2 text-xs sm:text-sm text-[#141312]/70 leading-relaxed">
                  Every weight, tokenizer rule, and training dataset is open and inspectable—no
                  hidden black boxes.
                </p>
              </div>

              <div>
                <span className="font-mono-tabular text-xs text-[#FF5500] uppercase tracking-widest">
                  02 / Honest Answers
                </span>
                <h3 className="font-display text-lg font-bold text-[#141312] mt-1">
                  Zero Guessing
                </h3>
                <p className="mt-2 text-xs sm:text-sm text-[#141312]/70 leading-relaxed">
                  Helios checks whether a topic is grounded in its knowledge before answering, rather
                  than fabricating facts.
                </p>
              </div>

              <div>
                <span className="font-mono-tabular text-xs text-[#FF5500] uppercase tracking-widest">
                  03 / Local Hardware
                </span>
                <h3 className="font-display text-lg font-bold text-[#141312] mt-1">
                  8× Lighter Memory
                </h3>
                <p className="mt-2 text-xs sm:text-sm text-[#141312]/70 leading-relaxed">
                  Grouped-Query Attention cuts memory usage so even the 5.80B Flagship runs on a
                  standard 16GB laptop.
                </p>
              </div>
            </div>
          </div>

          {/* Right 5 Columns: Frameless Sculpture Cutout + Sun Aura Overlay */}
          <div className="lg:col-span-5 relative flex items-center justify-center">
            <div className="pointer-events-none absolute w-72 h-72 sm:w-96 sm:h-96 rounded-full bg-gradient-to-tr from-[#FF5500]/20 via-[#FF8800]/10 to-transparent blur-2xl" />

            <div className="pointer-events-none absolute -top-6 right-4 w-28 h-28 sm:w-36 sm:h-36 rounded-full overflow-hidden opacity-85 mix-blend-multiply shadow-lg">
              <img
                src="/assets/helios_sun.jpeg"
                alt="Classical Sun Emblem"
                className="w-full h-full object-cover"
              />
            </div>

            <div className="relative z-10">
              <img
                src="/assets/ancient-greek-style-female-statue-isolated-on-transparent-background-png.webp"
                alt="Classical Greek Sculpture"
                className="w-64 sm:w-80 md:w-96 h-auto object-contain mx-auto drop-shadow-[0_24px_45px_rgba(20,19,18,0.22)]"
              />
            </div>
          </div>
        </div>
      </div>
    </ScrollZoomSection>
  );
};
