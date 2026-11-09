import React, { useState } from 'react';
import { Check, Copy, ArrowUpRight, Code2, Cpu, Wrench, BookOpen } from 'lucide-react';
import { OllamaLogo, HuggingFaceLogo, OpenClawLogo, HermesAgentLogo } from '../components/PlatformLogos';
import { KineticHeading } from '../components/KineticText';

interface PlatformSnippet {
  id: string;
  name: string;
  icon: React.ElementType;
  language: string;
  description: string;
  command: string;
}

const PLATFORM_SNIPPETS: PlatformSnippet[] = [
  {
    id: 'ollama',
    name: 'Ollama CLI',
    icon: OllamaLogo,
    language: 'bash',
    description: 'Pull and run quantized Helios GGUF weights directly from terminal.',
    command: `# Pull and run the Flagship 5.80B model locally
ollama run parejasarronkian/Helios:flagship

# Or run the ultra-edge Nano model for instant intent routing
ollama run parejasarronkian/Helios:nano`,
  },
  {
    id: 'openclaw',
    name: 'OpenClaw Agent',
    icon: OpenClawLogo,
    language: 'json',
    description: 'Configure Helios as the reasoning engine for OpenClaw autonomous tool execution.',
    command: `{
  "agent": {
    "name": "helios-claw-worker",
    "provider": "ollama",
    "model": "parejasarronkian/Helios:flagship",
    "temperature": 0.1,
    "system_prompt": "You are Helios, executing multi-turn tool loops under OpenClaw.",
    "tools": ["filesystem", "bash_sandbox", "browser_view", "mcp_runner"]
  }
}`,
  },
  {
    id: 'hermes',
    name: 'Hermes Agent',
    icon: HermesAgentLogo,
    language: 'python',
    description: 'Run Helios within the Nous Research Hermes function calling & autonomous agent loop.',
    command: `from hermes_agent import HermesRunner

# Initialize autonomous Hermes reasoning loop powered by Helios
runner = HermesRunner(
    model_name="parejasarronkian/Helios:flagship",
    backend="ollama", # or vLLM / HuggingFace
    function_calling_format="hermes-3-xml",
    enable_scratchpad=True
)

response = runner.run("Audit repo dependencies, plan migration, and commit fixes.")
print(response.final_output)`,
  },
  {
    id: 'python',
    name: 'Python / Transformers',
    icon: Code2,
    language: 'python',
    description: 'Load SafeTensors weights via Hugging Face Transformers with BF16 precision.',
    command: `import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Darknecrocities/Helios-Flagship-6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

inputs = tokenizer("State the first principles of Helios:", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))`,
  },
  {
    id: 'llamacpp',
    name: 'llama.cpp (GGUF)',
    icon: Cpu,
    language: 'bash',
    description: 'Direct quantized CPU/Metal inference with zero external dependencies.',
    command: `# Run native Q4_K_M quantized weights with Metal acceleration
./llama-cli \\
  -m ./models/helios-flagship-q4_k_m.gguf \\
  -p "Explain the 8x Grouped-Query Attention mechanism:" \\
  -c 32768 \\
  --temp 0.2`,
  },
  {
    id: 'mcp',
    name: 'Antigravity MCP Agent',
    icon: Wrench,
    language: 'json',
    description: 'Register Helios as an autonomous agent reasoning provider.',
    command: `{
  "mcpServers": {
    "helios-local": {
      "command": "ollama",
      "args": ["run", "parejasarronkian/Helios:flagship"],
      "env": {
        "HELIOS_CONTEXT_WINDOW": "262144",
        "HELIOS_MEMORY_TIER": "ALL"
      }
    }
  }
}`,
  },
];

interface PlatformsSectionProps {
  onOpenDocs: () => void;
}

export const PlatformsSection: React.FC<PlatformsSectionProps> = ({ onOpenDocs }) => {
  const [activeTabId, setActiveTabId] = useState<string>('ollama');
  const [copied, setCopied] = useState<boolean>(false);

  const activeSnippet = PLATFORM_SNIPPETS.find((s) => s.id === activeTabId) ?? PLATFORM_SNIPPETS[0];

  const handleCopy = () => {
    navigator.clipboard.writeText(activeSnippet.command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="platforms" className="py-20 md:py-28 bg-[#FBF9F5] text-[#141312] border-t border-[#141312]/8">
      <div className="max-w-6xl mx-auto px-4 sm:px-6">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between pb-8 border-b border-[#141312]/10 gap-4">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#141312]/5 border border-[#141312]/10 text-[10px] font-mono-tabular uppercase tracking-widest text-[#FF5500] mb-2">
              <span>DEVELOPER DEPLOYMENT HUB</span>
            </div>
            <KineticHeading
              text="RUN HELIOS ANYWHERE"
              highlightWords={['HELIOS']}
              className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-[#141312]"
            />
          </div>
          <div>
            <p className="text-xs sm:text-sm text-[#141312]/70 max-w-md leading-relaxed">
              Standard open weights compatible with Ollama, Hugging Face, llama.cpp, vLLM, and Model Context Protocol (MCP).
            </p>
            <button
              onClick={onOpenDocs}
              className="mt-3 inline-flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-[#141312] hover:bg-[#FF5500] text-white text-xs font-mono-tabular transition-colors cursor-pointer"
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>Explore Complete Docs & Datasets</span>
            </button>
          </div>
        </div>

        {/* Devlets Multi-Environment Developer Terminal */}
        <div className="mt-8 rounded-2xl bg-[#141312] text-[#FAF7F2] border border-white/10 shadow-2xl overflow-hidden">
          {/* Terminal Tab Bar */}
          <div className="px-4 py-3 bg-[#0D0C0B] border-b border-white/10 flex flex-wrap items-center justify-between gap-3">
            {/* Tabs */}
            <div className="flex flex-wrap items-center gap-1.5">
              {PLATFORM_SNIPPETS.map((snippet) => {
                const isActive = snippet.id === activeTabId;
                const Icon = snippet.icon;
                return (
                  <button
                    key={snippet.id}
                    onClick={() => setActiveTabId(snippet.id)}
                    className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-mono-tabular transition-all cursor-pointer ${
                      isActive
                        ? 'bg-[#FF5500] text-white font-medium shadow-xs'
                        : 'text-white/60 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    <Icon size={14} className="shrink-0" />
                    <span>{snippet.name}</span>
                  </button>
                );
              })}
            </div>

            {/* Copy Button */}
            <button
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/20 text-white text-xs font-mono-tabular transition-colors cursor-pointer active:scale-95"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-[#FF5500]" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy Code'}</span>
            </button>
          </div>

          {/* Description Strip */}
          <div className="px-6 py-3 bg-white/[0.02] border-b border-white/5 text-xs text-white/70 font-mono-tabular flex items-center justify-between">
            <span>{activeSnippet.description}</span>
            <span className="text-[#FF5500] uppercase text-[10px]">Lang: {activeSnippet.language}</span>
          </div>

          {/* Code Viewer Stage */}
          <div className="p-6 overflow-x-auto font-mono-tabular text-xs text-white/90 leading-relaxed bg-black/60 min-h-[160px]">
            <pre>
              <code>{activeSnippet.command}</code>
            </pre>
          </div>
        </div>

        {/* Direct Platform Links */}
        <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <a
            href="https://ollama.com/parejasarronkian"
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 rounded-2xl bg-white border border-[#141312]/10 hover:border-[#FF5500]/50 shadow-xs hover:shadow-md transition-all flex items-center justify-between group"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-[#141312] text-white">
                <OllamaLogo size={22} />
              </div>
              <div className="min-w-0">
                <h4 className="font-display font-bold text-xs text-[#141312] truncate">
                  Ollama Registry
                </h4>
                <p className="text-[11px] text-[#141312]/60 font-mono-tabular truncate">
                  parejasarronkian
                </p>
              </div>
            </div>
            <ArrowUpRight className="w-4 h-4 text-[#141312]/40 group-hover:text-[#FF5500] group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all shrink-0" />
          </a>

          <a
            href="https://huggingface.co/Darknecrocities/Helios"
            target="_blank"
            rel="noopener noreferrer"
            className="p-4 rounded-2xl bg-white border border-[#141312]/10 hover:border-[#FF5500]/50 shadow-xs hover:shadow-md transition-all flex items-center justify-between group"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-[#141312] text-white">
                <HuggingFaceLogo size={22} />
              </div>
              <div className="min-w-0">
                <h4 className="font-display font-bold text-xs text-[#141312] truncate">
                  Hugging Face
                </h4>
                <p className="text-[11px] text-[#141312]/60 font-mono-tabular truncate">
                  Darknecrocities
                </p>
              </div>
            </div>
            <ArrowUpRight className="w-4 h-4 text-[#141312]/40 group-hover:text-[#FF5500] group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all shrink-0" />
          </a>

          <button
            onClick={() => {
              setActiveTabId('openclaw');
              const el = document.getElementById('platforms');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }}
            className="p-4 rounded-2xl bg-white border border-[#141312]/10 hover:border-[#FF5500]/50 shadow-xs hover:shadow-md transition-all flex items-center justify-between group text-left cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-[#141312] text-white">
                <OpenClawLogo size={22} />
              </div>
              <div className="min-w-0">
                <h4 className="font-display font-bold text-xs text-[#141312] truncate">
                  OpenClaw Agent
                </h4>
                <p className="text-[11px] text-[#141312]/60 font-mono-tabular truncate">
                  Cybernetic Tool Loop
                </p>
              </div>
            </div>
            <ArrowUpRight className="w-4 h-4 text-[#141312]/40 group-hover:text-[#FF5500] group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all shrink-0" />
          </button>

          <button
            onClick={() => {
              setActiveTabId('hermes');
              const el = document.getElementById('platforms');
              if (el) el.scrollIntoView({ behavior: 'smooth' });
            }}
            className="p-4 rounded-2xl bg-white border border-[#141312]/10 hover:border-[#FF5500]/50 shadow-xs hover:shadow-md transition-all flex items-center justify-between group text-left cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-[#141312] text-white">
                <HermesAgentLogo size={22} />
              </div>
              <div className="min-w-0">
                <h4 className="font-display font-bold text-xs text-[#141312] truncate">
                  Hermes Agent
                </h4>
                <p className="text-[11px] text-[#141312]/60 font-mono-tabular truncate">
                  Nous Function Loop
                </p>
              </div>
            </div>
            <ArrowUpRight className="w-4 h-4 text-[#141312]/40 group-hover:text-[#FF5500] group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-all shrink-0" />
          </button>
        </div>
      </div>
    </section>
  );
};
