import React, { useState, useRef } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Volume2,
  VolumeX,
  Maximize2,
  Terminal,
  Cpu,
  Layers,
  CheckCircle2,
} from 'lucide-react';

interface DemoVideoOption {
  id: string;
  title: string;
  tag: string;
  src: string;
  desc: string;
  duration: string;
  badge: string;
}

const DEMO_VIDEOS: DemoVideoOption[] = [
  {
    id: 'overview',
    title: 'Helios Core via Ollama',
    tag: 'Full Walkthrough',
    src: '/assets/video/helios_ollama_overview.mp4',
    desc: 'Real-time Ollama interface generation: geography, Filipino context, and Python scripts on local hardware.',
    duration: '1:57',
    badge: 'parejasarronkian/Helios',
  },
  {
    id: 'reasoning',
    title: 'Complex Math & Reasoning',
    tag: 'Deliberative Thought',
    src: '/assets/video/helios_ollama_reasoning.mp4',
    desc: 'Multi-step mathematical proof decomposition solving irrational number exponent equations from first principles.',
    duration: '0:42',
    badge: 'Hermes-3 XML · <thought>',
  },
  {
    id: 'conversation',
    title: 'Open Domain Knowledge',
    tag: 'Zero Hallucination',
    src: '/assets/video/helios_ollama_conversation.mp4',
    desc: 'Epistemic honesty boundary testing: identifying verified knowledge limits without inventing false facts.',
    duration: '1:03',
    badge: '100% Offline',
  },
];

export const ModelDemoSection: React.FC = () => {
  const [activeVideoId, setActiveVideoId] = useState<string>('overview');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [isMuted, setIsMuted] = useState<boolean>(true);
  const videoRef = useRef<HTMLVideoElement>(null);

  const activeVideo = DEMO_VIDEOS.find((v) => v.id === activeVideoId) ?? DEMO_VIDEOS[0];

  const handleSelectVideo = (id: string) => {
    setActiveVideoId(id);
    setIsPlaying(true);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play().catch(() => {});
    }
  };

  const togglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().catch(() => {});
      setIsPlaying(true);
    }
  };

  const toggleMute = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleRestart = () => {
    if (!videoRef.current) return;
    videoRef.current.currentTime = 0;
    videoRef.current.play().catch(() => {});
    setIsPlaying(true);
  };

  const handleFullscreen = () => {
    if (!videoRef.current) return;
    if (videoRef.current.requestFullscreen) {
      videoRef.current.requestFullscreen();
    }
  };

  return (
    <section id="demo" className="py-24 sm:py-32 bg-[#FAF7F2] border-t border-[#141312]/8 relative overflow-hidden">
      {/* Subtle background ambient gradients */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] bg-gradient-to-b from-[#FF5500]/5 to-transparent rounded-full blur-3xl pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div className="text-center max-w-3xl mx-auto space-y-4 mb-12 sm:mb-16">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white border border-[#141312]/10 shadow-xs">
            <span className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse" />
            <span className="font-mono-tabular text-[10px] uppercase tracking-[0.25em] text-[#141312]/70">
              Live Model Execution · Ollama Local Host
            </span>
          </div>

          <h2 className="font-display font-bold text-3xl sm:text-4xl md:text-5xl text-[#141312] tracking-tight">
            See Helios Running Live on Everyday Hardware.
          </h2>

          <p className="font-serif-editorial text-base sm:text-lg text-[#141312]/70 max-w-2xl mx-auto leading-relaxed">
            Real screen capture of Helios executing locally through the Ollama runtime with sub-14ms step latency, zero cloud APIs, and full epistemic verification.
          </p>

          {/* Interactive Scenario Tabs */}
          <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
            {DEMO_VIDEOS.map((demo) => {
              const isActive = demo.id === activeVideoId;
              return (
                <button
                  key={demo.id}
                  onClick={() => handleSelectVideo(demo.id)}
                  className={`px-3.5 py-1.5 rounded-full text-xs font-mono-tabular transition-all cursor-pointer flex items-center gap-2 ${
                    isActive
                      ? 'bg-[#141312] text-white shadow-sm font-medium scale-[1.02]'
                      : 'bg-white/80 text-[#141312]/70 hover:bg-white border border-[#141312]/10 hover:text-[#141312]'
                  }`}
                >
                  <span>{demo.title}</span>
                  <span className={`text-[10px] px-1.5 py-0.2 rounded-full ${isActive ? 'bg-[#FF5500] text-white' : 'bg-black/5 text-black/50'}`}>
                    {demo.duration}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Realistic Laptop Mockup Frame */}
        <div className="max-w-5xl mx-auto">
          {/* Laptop Screen Upper Chassis */}
          <div className="relative rounded-t-[28px] sm:rounded-t-[34px] bg-[#1C1A18] p-3 sm:p-5 shadow-[0_30px_90px_rgba(20,19,18,0.22)] border border-[#3A3734] border-b-0">
            {/* Camera Notch and Ambient Sensor */}
            <div className="absolute top-2.5 sm:top-3.5 left-1/2 -translate-x-1/2 flex items-center gap-1.5 z-20">
              <div className="w-1.5 h-1.5 rounded-full bg-[#0D0C0B] ring-1 ring-white/10" />
              <div className="w-1 h-1 rounded-full bg-[#10B981]/80 ring-1 ring-[#10B981]/30" />
            </div>

            {/* Display Bezel Interior Frame */}
            <div className="relative rounded-t-xl sm:rounded-t-2xl overflow-hidden bg-[#0D0C0B] border border-white/5 aspect-[16/10] sm:aspect-[16/10.5]">
              {/* Window Title Bar */}
              <div className="h-8 sm:h-9 bg-[#171615] border-b border-white/5 px-3 sm:px-4 flex items-center justify-between text-xs text-white/50 select-none">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#FF5F56] border border-[#E0443E]/50" />
                  <span className="w-2.5 h-2.5 rounded-full bg-[#FFBD2E] border border-[#DEA123]/50" />
                  <span className="w-2.5 h-2.5 rounded-full bg-[#27C93F] border border-[#1AAB29]/50" />
                </div>

                <div className="flex items-center gap-1.5 font-mono-tabular text-[10px] sm:text-[11px] text-white/70">
                  <Terminal className="w-3 h-3 text-[#FF5500]" />
                  <span>ollama run {activeVideo.badge}</span>
                </div>

                <div className="flex items-center gap-2 font-mono-tabular text-[9px] sm:text-[10px] text-[#10B981]">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#10B981] animate-ping" />
                  <span>LOCAL MPS (60 FPS)</span>
                </div>
              </div>

              {/* Video Player */}
              <div className="relative w-full h-[calc(100%-2rem)] sm:h-[calc(100%-2.25rem)] bg-black">
                <video
                  ref={videoRef}
                  key={activeVideo.src}
                  src={activeVideo.src}
                  poster="/assets/video/helios_demo_poster.jpg"
                  autoPlay
                  loop
                  muted={isMuted}
                  playsInline
                  className="w-full h-full object-contain bg-black"
                  onPlay={() => setIsPlaying(true)}
                  onPause={() => setIsPlaying(false)}
                />

                {/* Overlay Playback Controls on Hover / Touch */}
                <div className="absolute bottom-3 left-3 right-3 sm:bottom-4 sm:left-4 sm:right-4 flex items-center justify-between gap-2 p-2 sm:p-2.5 rounded-xl bg-black/70 backdrop-blur-md border border-white/10 text-white opacity-90 transition-opacity">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={togglePlay}
                      className="p-1.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors text-white cursor-pointer"
                      aria-label={isPlaying ? 'Pause' : 'Play'}
                    >
                      {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                    </button>

                    <button
                      onClick={handleRestart}
                      className="p-1.5 rounded-lg bg-white/5 hover:bg-white/15 transition-colors text-white/80 cursor-pointer"
                      aria-label="Restart Video"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>

                    <button
                      onClick={toggleMute}
                      className="p-1.5 rounded-lg bg-white/5 hover:bg-white/15 transition-colors text-white/80 cursor-pointer"
                      aria-label={isMuted ? 'Unmute' : 'Mute'}
                    >
                      {isMuted ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
                    </button>

                    <span className="font-mono-tabular text-[10px] sm:text-xs text-white/70 ml-1">
                      {activeVideo.title}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="hidden sm:inline-block px-2 py-0.5 rounded text-[9px] font-mono-tabular bg-[#FF5500]/20 text-[#FF5500] border border-[#FF5500]/30">
                      {activeVideo.tag}
                    </span>
                    <button
                      onClick={handleFullscreen}
                      className="p-1.5 rounded-lg bg-white/5 hover:bg-white/15 transition-colors text-white/80 cursor-pointer"
                      aria-label="Fullscreen"
                    >
                      <Maximize2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Laptop Base (Hinge & Keyboard Deck Top) */}
          <div className="relative">
            {/* Recessed Screen Hinge */}
            <div className="mx-auto w-40 sm:w-56 h-2 sm:h-2.5 bg-[#121110] rounded-b-md border-x border-b border-[#2C2A28]" />

            {/* Aluminum Keyboard Deck Front Plate */}
            <div className="relative h-4 sm:h-5 bg-gradient-to-b from-[#2E2C2A] via-[#242220] to-[#181716] rounded-b-2xl sm:rounded-b-3xl shadow-[0_20px_40px_rgba(0,0,0,0.3)] border-t border-white/10 flex items-center justify-center">
              {/* Display Opening Thumb Groove */}
              <div className="w-16 sm:w-24 h-1 sm:h-1.5 bg-[#141312] rounded-full opacity-60" />
            </div>

            {/* Subtle Surface Reflection / Desk Shadow */}
            <div className="mx-auto w-[92%] h-4 bg-gradient-to-b from-black/25 to-transparent blur-md -mt-1 rounded-full pointer-events-none" />
          </div>

          {/* Feature Badges Under Laptop */}
          <div className="mt-10 grid grid-cols-1 sm:grid-cols-3 gap-4 pt-4 border-t border-[#141312]/8 text-left">
            <div className="p-3.5 rounded-xl bg-white/60 border border-[#141312]/5 flex items-start gap-3">
              <Cpu className="w-4 h-4 text-[#FF5500] shrink-0 mt-0.5" />
              <div>
                <div className="font-display text-xs font-bold text-[#141312]">Apple Silicon Metal (MPS)</div>
                <div className="text-[11px] text-[#141312]/60 mt-0.5">Runs at full speed natively with unified memory and zero lag.</div>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-white/60 border border-[#141312]/5 flex items-start gap-3">
              <Layers className="w-4 h-4 text-[#FF5500] shrink-0 mt-0.5" />
              <div>
                <div className="font-display text-xs font-bold text-[#141312]">Standard 1-Command Pull</div>
                <div className="text-[11px] text-[#141312]/60 mt-0.5"><code className="font-mono-tabular">ollama run parejasarronkian/Helios</code></div>
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-white/60 border border-[#141312]/5 flex items-start gap-3">
              <CheckCircle2 className="w-4 h-4 text-[#FF5500] shrink-0 mt-0.5" />
              <div>
                <div className="font-display text-xs font-bold text-[#141312]">100% Private & Offline</div>
                <div className="text-[11px] text-[#141312]/60 mt-0.5">Zero tokens sent to external servers. Your prompts never leave your device.</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};
