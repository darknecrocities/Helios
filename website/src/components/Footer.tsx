import React from 'react';
import { ArrowUpRight, Mail } from 'lucide-react';
import { OllamaLogo, HuggingFaceLogo } from './PlatformLogos';

interface FooterProps {
  onOpenDocs: () => void;
  onNavigateSection: (sectionId: string) => void;
}

export const Footer: React.FC<FooterProps> = ({ onOpenDocs, onNavigateSection }) => {
  return (
    <footer className="bg-[#141312] text-[#FAF7F2] py-16 border-t border-white/10">
      <div className="max-w-[1320px] mx-auto px-4 sm:px-8 md:px-12">
        <div className="grid grid-cols-1 md:grid-cols-12 gap-10 pb-12 border-b border-white/10">
          {/* Left: Brand & Architect */}
          <div className="md:col-span-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full overflow-hidden border border-[#FF5500]/60 bg-[#141312] shrink-0">
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

            <p className="text-sm text-[#FAF7F2]/70 max-w-md leading-relaxed">
              Open-weight foundation model family conceived, architected, and built by{' '}
              <strong className="text-white">Arron Kian Parejas</strong> in the Philippines.
            </p>

            <a
              href="mailto:parejasarronkian@gmail.com"
              className="inline-flex items-center gap-2 text-xs font-mono-tabular text-[#FF5500] hover:underline"
            >
              <Mail className="w-3.5 h-3.5" />
              <span>parejasarronkian@gmail.com</span>
            </a>
          </div>

          {/* Center: Quick Navigation */}
          <div className="md:col-span-3 space-y-2.5 text-xs">
            <span className="uppercase tracking-widest text-[#FAF7F2]/45 block mb-3">
              Navigation
            </span>
            <ul className="space-y-2 text-[#FAF7F2]/75">
              <li>
                <button
                  onClick={() => onNavigateSection('about')}
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  About Helios
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigateSection('family')}
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Model Family
                </button>
              </li>
              <li>
                <button
                  onClick={() => onNavigateSection('comparison')}
                  className="hover:text-white transition-colors cursor-pointer"
                >
                  Model Comparison
                </button>
              </li>
              <li>
                <button
                  onClick={onOpenDocs}
                  className="text-[#FF5500] font-medium hover:underline cursor-pointer"
                >
                  Documentation & Datasets
                </button>
              </li>
            </ul>
          </div>

          {/* Right: Platforms */}
          <div className="md:col-span-3 space-y-2.5 text-xs">
            <span className="uppercase tracking-widest text-[#FAF7F2]/45 block mb-3">
              Platforms
            </span>
            <div className="flex flex-col gap-2">
              <a
                href="https://ollama.com/parejasarronkian"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-[#FAF7F2] transition-colors"
              >
                <span className="flex items-center gap-2">
                  <OllamaLogo size={14} />
                  <span>Ollama · parejasarronkian/Helios</span>
                </span>
                <ArrowUpRight className="w-3.5 h-3.5 text-[#FF5500]" />
              </a>

              <a
                href="https://huggingface.co/Darknecrocities/Helios"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-[#FAF7F2] transition-colors"
              >
                <span className="flex items-center gap-2">
                  <HuggingFaceLogo size={14} />
                  <span>HF · Darknecrocities/Helios</span>
                </span>
                <ArrowUpRight className="w-3.5 h-3.5 text-[#FF5500]" />
              </a>
            </div>
          </div>
        </div>

        <div className="pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-[#FAF7F2]/45">
          <span>© {new Date().getFullYear()} Helios · Arron Kian Parejas</span>
          <span>Apache 2.0 Open Weights</span>
        </div>
      </div>
    </footer>
  );
};
