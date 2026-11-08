import React, { useState } from 'react';
import { Check, Copy, ArrowUpRight } from 'lucide-react';
import { OllamaLogo, HuggingFaceLogo } from '../components/PlatformLogos';
import { MorphingCard } from '../components/MorphingCard';
import { KineticHeading } from '../components/KineticText';
import { ScrollZoomSection } from '../components/ScrollZoomSection';

interface PlatformsSectionProps {
  onOpenDocs: () => void;
}

export const PlatformsSection: React.FC<PlatformsSectionProps> = ({ onOpenDocs }) => {
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const handleCopy = (cmd: string) => {
    navigator.clipboard.writeText(cmd);
    setCopiedCmd(cmd);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  return (
    <ScrollZoomSection
      id="platforms"
      className="py-28 md:py-36 bg-[#F5F0E6] text-[#141312] border-t border-[#141312]/10"
    >
      <div className="max-w-[1320px] mx-auto px-4 sm:px-8 md:px-12 relative z-10">
        <div className="flex flex-col md:flex-row md:items-end justify-between pb-10 border-b border-[#141312]/10 gap-6">
          <div>
            <KineticHeading
              text="RUN HELIOS ANYWHERE"
              highlightWords={['HELIOS']}
              className="font-display text-3xl sm:text-5xl font-bold tracking-tight text-[#141312]"
            />
          </div>
          <p className="text-sm sm:text-base text-[#141312]/70 max-w-md leading-relaxed">
            Available on Ollama and Hugging Face with native support for GGUF, SafeTensors, and
            ONNX.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-10">
          {/* Ollama Card — Pure Brand Logo Without Background Box */}
          <MorphingCard className="p-8 sm:p-10 bg-[#FAF7F2] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <OllamaLogo size={54} className="text-[#141312]" />
                <a
                  href="https://ollama.com/parejasarronkian"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-xs uppercase tracking-wider text-[#141312]/70 hover:text-[#FF5500] transition-colors"
                >
                  <span>ollama.com/parejasarronkian</span>
                  <ArrowUpRight className="w-4 h-4" />
                </a>
              </div>

              <h3 className="font-display text-2xl font-bold text-[#141312] mt-6">
                Ollama · parejasarronkian/Helios
              </h3>
              <p className="mt-2 text-sm text-[#141312]/70 leading-relaxed">
                Pull and run quantized Helios GGUF weights (<code className="font-mono text-xs">:nano</code>, <code className="font-mono text-xs">:alpha</code>, <code className="font-mono text-xs">:beta</code>, <code className="font-mono text-xs">:core</code>, <code className="font-mono text-xs">:pro</code>, <code className="font-mono text-xs">:flagship</code>) locally on macOS, Linux, or Windows with a single command.
              </p>
            </div>

            <div className="mt-8 pt-5 border-t border-[#141312]/10 flex items-center justify-between gap-3">
              <code className="font-mono-tabular text-xs sm:text-sm text-[#141312]">
                ollama run parejasarronkian/Helios:flagship
              </code>
              <button
                onClick={() => handleCopy('ollama run parejasarronkian/Helios:flagship')}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-[#141312] hover:bg-[#FF5500] text-[#FAF7F2] text-xs transition-colors cursor-pointer"
              >
                {copiedCmd === 'ollama run parejasarronkian/Helios:flagship' ? (
                  <Check className="w-3.5 h-3.5" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
                <span>Copy</span>
              </button>
            </div>
          </MorphingCard>

          {/* Hugging Face Card — Pure Brand Logo Without Background Box */}
          <MorphingCard className="p-8 sm:p-10 bg-[#FAF7F2] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <HuggingFaceLogo size={54} />
                <a
                  href="https://huggingface.co/Darknecrocities/Helios"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 text-xs uppercase tracking-wider text-[#141312]/70 hover:text-[#FF5500] transition-colors"
                >
                  <span>huggingface.co/Darknecrocities/Helios</span>
                  <ArrowUpRight className="w-4 h-4" />
                </a>
              </div>

              <h3 className="font-display text-2xl font-bold text-[#141312] mt-6">
                Hugging Face · Darknecrocities/Helios
              </h3>
              <p className="mt-2 text-sm text-[#141312]/70 leading-relaxed">
                Download open-weight SafeTensors checkpoints, BPE tokenizers, LoRA adapters, and
                full training configurations.
              </p>
            </div>

            <div className="mt-8 pt-5 border-t border-[#141312]/10 flex items-center justify-between gap-3">
              <code className="font-mono-tabular text-xs sm:text-sm text-[#141312] truncate">
                huggingface-cli download Darknecrocities/Helios
              </code>
              <button
                onClick={() => handleCopy('huggingface-cli download Darknecrocities/Helios')}
                className="inline-flex items-center gap-1.5 px-4 py-2 rounded-full bg-[#141312] hover:bg-[#FF5500] text-[#FAF7F2] text-xs transition-colors shrink-0 cursor-pointer"
              >
                {copiedCmd === 'huggingface-cli download Darknecrocities/Helios' ? (
                  <Check className="w-3.5 h-3.5" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
                <span>Copy</span>
              </button>
            </div>
          </MorphingCard>
        </div>

        {/* Minimalist Documentation Invitation */}
        <div className="mt-12 p-8 sm:p-12 rounded-3xl bg-[#141312] text-[#FAF7F2] flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div>
            <h3 className="font-display text-2xl sm:text-3xl font-bold">
              Documentation, Architecture & All 177 Pretraining Datasets
            </h3>
            <p className="mt-2 text-sm text-[#FAF7F2]/70 max-w-xl">
              Read the complete guide covering how Helios works, how to fine-tune it with LoRA, and
              explore the full list of 177 datasets used for pretraining.
            </p>
          </div>
          <button
            onClick={() => {
              onOpenDocs();
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className="px-7 py-4 rounded-full bg-[#FF5500] hover:bg-[#E64A00] text-white text-xs font-semibold uppercase tracking-[0.18em] transition-colors shrink-0 cursor-pointer"
          >
            Open Documentation
          </button>
        </div>
      </div>
    </ScrollZoomSection>
  );
};
