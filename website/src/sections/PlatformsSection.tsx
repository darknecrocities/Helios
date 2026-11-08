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
      className="py-16 md:py-22 bg-[#F5F0E6] text-[#141312] border-t border-[#141312]/10"
    >
      <div className="max-w-5xl mx-auto px-4 sm:px-6 relative z-10">
        <div className="flex flex-col md:flex-row md:items-end justify-between pb-7 border-b border-[#141312]/10 gap-4">
          <div>
            <KineticHeading
              text="RUN HELIOS ANYWHERE"
              highlightWords={['HELIOS']}
              className="font-display text-2xl sm:text-4xl font-bold tracking-tight text-[#141312]"
            />
          </div>
          <p className="text-xs text-[#141312]/70 max-w-sm leading-relaxed">
            Available on Ollama and Hugging Face with native support for GGUF, SafeTensors, and
            ONNX.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-7">
          {/* Compact Ollama Card */}
          <MorphingCard className="p-5 sm:p-6 rounded-2xl bg-[#FAF7F2] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <OllamaLogo size={34} className="text-[#141312]" />
                <a
                  href="https://ollama.com/parejasarronkian"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-[#141312]/70 hover:text-[#FF5500] transition-colors"
                >
                  <span>ollama.com/parejasarronkian</span>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </a>
              </div>

              <h3 className="font-display text-lg font-bold text-[#141312] mt-4">
                Ollama · parejasarronkian/Helios
              </h3>
              <p className="mt-1.5 text-xs text-[#141312]/70 leading-relaxed">
                Pull and run quantized Helios GGUF weights (<code className="font-mono text-[11px]">:nano</code>, <code className="font-mono text-[11px]">:alpha</code>, <code className="font-mono text-[11px]">:beta</code>, <code className="font-mono text-[11px]">:core</code>, <code className="font-mono text-[11px]">:pro</code>, <code className="font-mono text-[11px]">:flagship</code>) locally on macOS, Linux, or Windows.
              </p>
            </div>

            <div className="mt-5 pt-3.5 border-t border-[#141312]/10 flex items-center justify-between gap-2">
              <code className="font-mono-tabular text-xs text-[#141312] truncate">
                ollama run parejasarronkian/Helios:flagship
              </code>
              <button
                onClick={() => handleCopy('ollama run parejasarronkian/Helios:flagship')}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full bg-[#141312] hover:bg-[#FF5500] text-[#FAF7F2] text-[11px] transition-colors shrink-0 cursor-pointer"
              >
                {copiedCmd === 'ollama run parejasarronkian/Helios:flagship' ? (
                  <Check className="w-3 h-3" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
                <span>Copy</span>
              </button>
            </div>
          </MorphingCard>

          {/* Compact Hugging Face Card */}
          <MorphingCard className="p-5 sm:p-6 rounded-2xl bg-[#FAF7F2] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <HuggingFaceLogo size={34} />
                <a
                  href="https://huggingface.co/Darknecrocities/Helios"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-[10px] uppercase tracking-wider text-[#141312]/70 hover:text-[#FF5500] transition-colors"
                >
                  <span>huggingface.co/Darknecrocities/Helios</span>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </a>
              </div>

              <h3 className="font-display text-lg font-bold text-[#141312] mt-4">
                Hugging Face · Darknecrocities/Helios
              </h3>
              <p className="mt-1.5 text-xs text-[#141312]/70 leading-relaxed">
                Download open-weight SafeTensors checkpoints, BPE tokenizers, LoRA adapters, and
                full training configurations.
              </p>
            </div>

            <div className="mt-5 pt-3.5 border-t border-[#141312]/10 flex items-center justify-between gap-2">
              <code className="font-mono-tabular text-xs text-[#141312] truncate">
                huggingface-cli download Darknecrocities/Helios
              </code>
              <button
                onClick={() => handleCopy('huggingface-cli download Darknecrocities/Helios')}
                className="inline-flex items-center gap-1 px-3 py-1.5 rounded-full bg-[#141312] hover:bg-[#FF5500] text-[#FAF7F2] text-[11px] transition-colors shrink-0 cursor-pointer"
              >
                {copiedCmd === 'huggingface-cli download Darknecrocities/Helios' ? (
                  <Check className="w-3 h-3" />
                ) : (
                  <Copy className="w-3 h-3" />
                )}
                <span>Copy</span>
              </button>
            </div>
          </MorphingCard>
        </div>

        {/* Compact Documentation Invitation Banner */}
        <div className="mt-7 p-5 sm:p-7 rounded-2xl bg-[#141312] text-[#FAF7F2] flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <h3 className="font-display text-lg sm:text-xl font-bold">
              Documentation, Engineering Journal & All 177 Pretraining Datasets
            </h3>
            <p className="mt-1 text-xs text-[#FAF7F2]/70 max-w-lg">
              Read Arron Kian Parejas’s complete engineering journal (Research → Problem → Solution),
              MCP & 4-Tier Memory guide, LoRA fine-tuning, and all 177 datasets.
            </p>
          </div>
          <button
            onClick={() => {
              onOpenDocs();
              window.scrollTo({ top: 0, behavior: 'smooth' });
            }}
            className="px-5 py-2.5 rounded-full bg-[#FF5500] hover:bg-[#E64A00] text-white text-[11px] font-semibold uppercase tracking-[0.16em] transition-colors shrink-0 cursor-pointer"
          >
            Open Documentation
          </button>
        </div>
      </div>
    </ScrollZoomSection>
  );
};
