import React from 'react';
import { motion } from 'framer-motion';
import {
  NvidiaLogo,
  AntigravityLogo,
  HuggingFaceLogo,
  OllamaLogo,
  PythonLogo,
  PyTorchLogo,
  PyTorchSafeTensorsLogo,
  FaissLogo,
  SqliteLogo,
  OnnxLogo,
} from './PlatformLogos';

const BRAND_ITEMS = [
  { name: 'NVIDIA CUDA', Logo: NvidiaLogo },
  { name: 'Google Antigravity', Logo: AntigravityLogo },
  { name: 'Hugging Face', Logo: HuggingFaceLogo },
  { name: 'Ollama', Logo: OllamaLogo },
  { name: 'Python', Logo: PythonLogo },
  { name: 'PyTorch', Logo: PyTorchLogo },
  { name: 'SafeTensors', Logo: PyTorchSafeTensorsLogo },
  { name: 'FAISS Vector', Logo: FaissLogo },
  { name: 'SQLite WAL', Logo: SqliteLogo },
  { name: 'ONNX Runtime', Logo: OnnxLogo },
];

export const BrandBelt: React.FC = () => {
  // Duplicate array 3 times for seamless infinite marquee loop
  const marqueeList = [...BRAND_ITEMS, ...BRAND_ITEMS, ...BRAND_ITEMS];

  return (
    <div className="relative w-full py-12 bg-[#F5F0E6] border-y border-[#141312]/10 overflow-hidden select-none">
      {/* Subtle left & right fade masks */}
      <div className="pointer-events-none absolute inset-y-0 left-0 w-24 sm:w-40 bg-gradient-to-r from-[#F5F0E6] to-transparent z-10" />
      <div className="pointer-events-none absolute inset-y-0 right-0 w-24 sm:w-40 bg-gradient-to-l from-[#F5F0E6] to-transparent z-10" />

      <div className="max-w-[1320px] mx-auto px-4 mb-6 text-center">
        <span className="font-mono-tabular text-[11px] uppercase tracking-[0.26em] text-[#141312]/50">
          ENGINEERED & POWERED BY OPEN ECOSYSTEM TECHNOLOGIES
        </span>
      </div>

      <div className="flex overflow-hidden">
        <motion.div
          animate={{ x: ['0%', '-33.333%'] }}
          transition={{
            duration: 32,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="flex items-center gap-14 sm:gap-20 whitespace-nowrap px-8"
        >
          {marqueeList.map((item, idx) => {
            const Icon = item.Logo;
            return (
              <div
                key={`${item.name}-${idx}`}
                className="flex items-center gap-3.5 text-[#141312]/80 hover:text-[#FF5500] transition-colors duration-300"
              >
                <Icon size={34} />
                <span className="font-display font-semibold text-sm sm:text-base tracking-[0.12em] uppercase">
                  {item.name}
                </span>
              </div>
            );
          })}
        </motion.div>
      </div>
    </div>
  );
};
