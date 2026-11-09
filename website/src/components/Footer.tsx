import React from 'react';
import { ArrowUpRight, Mail } from 'lucide-react';
import { OllamaLogo, HuggingFaceLogo } from './PlatformLogos';

interface FooterProps {
  onOpenDocs: () => void;
  onNavigateSection: (sectionId: string) => void;
}

export const Footer: React.FC<FooterProps> = ({ onOpenDocs, onNavigateSection }) => {
  return (
    <footer className="bg-[#0D0C0B] text-[#FAF7F2] py-16 border-t border-white/10">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-10 pb-12 border-b border-white/10">
          {/* Left: Brand & Architect */}
          <div className="md:col-span-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full overflow-hidden border border-[#FF5500]/60 bg-[#141312] shrink-0">
                <img
                  src="/assets/helios_logo.png"
                  alt="Helios"
                  className="w-full h-full object-cover"
                />
              </div>
              <span className="font-display font-bold tracking-[0.22em] text-base text-white">
                HELIOS
              </span>
            </div>

            <p className="text-xs sm:text-sm text-[#FAF7F2]/70 max-w-md leading-relaxed font-sans">
              An open-weight foundation model ecosystem engineered from first principles by{' '}
              <strong className="text-white font-medium">Arron Kian Parejas</strong> in the Philippines.
              Six tiers from 17.84M edge to 5.80B flagship intelligence.
            </p>

            <div className="pt-2 flex flex-wrap items-center gap-4 text-xs font-mono-tabular">
              <a
                href="mailto:parejasarronkian@gmail.com"
                className="inline-flex items-center gap-1.5 text-[#FF5500] hover:underline"
              >
                <Mail className="w-3.5 h-3.5" />
                <span>parejasarronkian@gmail.com</span>
              </a>
              <span className="text-white/30">·</span>
              <span className="text-white/50">Philippines</span>
            </div>
          </div>

          {/* Center: Navigation */}
          <div className="md:col-span-3 space-y-3 text-xs font-mono-tabular">
            <span className="uppercase tracking-widest text-[#FF5500] block text-[10px] font-bold">
              ARCHITECTURE
            </span>
            <ul className="space-y-2.5 text-white/70">
              <li>
                <button
                  onClick={() => onNavigateSection('about')}
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Philosophy & Axioms
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigateSection('family')}
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Model Family (6 Tiers)
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigateSection('capabilities')}
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Panoptes 4-Tier Memory
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigateSection('comparison')}
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Hardware Telemetry
                </button>
              </li>
              <li>
                <button
                  onClick={onOpenDocs}
                  className="text-[#FF5500] font-medium hover:underline cursor-pointer"
                >
                  Interactive Documentation
                </button>
              </li>
            </ul>
          </div>

          {/* Right: Ecosystem Registries */}
          <div className="md:col-span-3 space-y-3 text-xs font-mono-tabular">
            <span className="uppercase tracking-widest text-[#FF5500] block text-[10px] font-bold">
              REGISTRIES
            </span>
            <div className="flex flex-col gap-2">
              <a
                href="https://ollama.com/parejasarronkian"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-white transition-colors border border-white/5"
              >
                <span className="flex items-center gap-2">
                  <OllamaLogo size={14} />
                  <span>parejasarronkian/Helios</span>
                </span>
                <ArrowUpRight className="w-3.5 h-3.5 text-[#FF5500]" />
              </a>

              <a
                href="https://huggingface.co/Darknecrocities/Helios"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-white transition-colors border border-white/5"
              >
                <span className="flex items-center gap-2">
                  <HuggingFaceLogo size={14} />
                  <span>Darknecrocities/Helios</span>
                </span>
                <ArrowUpRight className="w-3.5 h-3.5 text-[#FF5500]" />
              </a>
            </div>
          </div>
        </div>

        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono-tabular text-white/40">
          <span>© {new Date().getFullYear()} Helios Model Ecosystem · Apache 2.0 Open Weights</span>
          <div className="flex items-center gap-1">
            <span>Conceived & Architected with</span>
            <span className="text-[#FF5500] font-bold">First Principles</span>
            <span>in PH</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
