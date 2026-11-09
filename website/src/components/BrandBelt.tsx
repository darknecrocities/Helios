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
  OpenClawLogo,
  HermesAgentLogo,
  GoogleColabLogo,
  JupyterLogo,
} from './PlatformLogos';

const BRAND_ITEMS = [
  { name: 'NVIDIA CUDA', Logo: NvidiaLogo },
  { name: 'Google Antigravity', Logo: AntigravityLogo },
  { name: 'Hugging Face', Logo: HuggingFaceLogo },
  { name: 'Ollama', Logo: OllamaLogo },
  { name: 'OpenClaw Agent', Logo: OpenClawLogo },
  { name: 'Hermes Agent', Logo: HermesAgentLogo },
  { name: 'Google Colab', Logo: GoogleColabLogo },
  { name: 'Jupyter Notebook', Logo: JupyterLogo },
  { name: 'Python', Logo: PythonLogo },
  { name: 'PyTorch', Logo: PyTorchLogo },
  { name: 'SafeTensors', Logo: PyTorchSafeTensorsLogo },
  { name: 'FAISS Vector', Logo: FaissLogo },
  { name: 'SQLite WAL', Logo: SqliteLogo },
  { name: 'ONNX Runtime', Logo: OnnxLogo },
];

export const BrandBelt: React.FC = () => {
  const marqueeList = [...BRAND_ITEMS, ...BRAND_ITEMS, ...BRAND_ITEMS];

  return (
    <div className="relative w-full py-10 bg-[#FBF9F5] border-y border-[#141312]/8 overflow-hidden select-none">
      {/* Subtle left & right gradient fade masks */}
      <div className="pointer-events-none absolute inset-y-0 left-0 w-20 sm:w-36 bg-gradient-to-r from-[#FBF9F5] to-transparent z-10" />
      <div className="pointer-events-none absolute inset-y-0 right-0 w-20 sm:w-36 bg-gradient-to-l from-[#FBF9F5] to-transparent z-10" />

      <div className="max-w-6xl mx-auto px-4 mb-5 text-center">
        <span className="font-mono-tabular text-[10px] uppercase tracking-[0.28em] text-[#141312]/50">
          ENGINEERED & VERIFIED ACROSS THE OPEN ECOSYSTEM
        </span>
      </div>

      <div className="flex overflow-hidden">
        <motion.div
          animate={{ x: ['0%', '-33.333%'] }}
          transition={{
            duration: 35,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="flex items-center gap-12 sm:gap-16 whitespace-nowrap px-6"
        >
          {marqueeList.map((item, idx) => {
            const Icon = item.Logo;
            return (
              <div
                key={`${item.name}-${idx}`}
                className="flex items-center gap-3 text-[#141312]/70 hover:text-[#FF5500] transition-colors duration-300 group cursor-default"
              >
                <Icon size={28} className="opacity-80 group-hover:opacity-100 transition-opacity" />
                <span className="font-display font-medium text-xs sm:text-sm tracking-[0.14em] uppercase">
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
