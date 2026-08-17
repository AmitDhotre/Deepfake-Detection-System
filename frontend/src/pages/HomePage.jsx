import React from 'react';
import { Radar, ScanFace, Cpu, Layers, ShieldCheck, ArrowRight, Waves, History } from 'lucide-react';

const FEATURES = [
  { icon: Waves, title: 'Forensic Signal Analysis', desc: 'Checks frequency artifacts, noise consistency, edge sharpness, and color mismatch. Works out of the box, no training required.' },
  { icon: Cpu, title: 'Trainable CNN Pipeline', desc: 'Drop in trained Xception or ResNet18 weights and the API automatically upgrades from heuristic to model-based scoring.' },
  { icon: Layers, title: 'Image & Video Support', desc: 'Frame-sampled video analysis with per-frame scoring and temporal variance, alongside single-image scans.' },
  { icon: History, title: 'Local Scan History', desc: 'Every scan is saved in your browser so you can revisit past results and track patterns over time.' },
];

const PIPELINE = [
  { step: '01', title: 'Heuristic Analyzer', desc: 'Signal-processing rules that run as a fallback whenever no trained model is loaded.' },
  { step: '02', title: 'Trained CNN', desc: 'ResNet18 or Xception, once you train and drop in weights.' },
  { step: '03', title: 'Auto-Selection', desc: 'The API picks the strongest available model automatically.' },
];

export default function HomePage({ onStartScan }) {
  return (
    <div className="w-full max-w-5xl mx-auto px-6 pb-24">
      {/* Hero */}
      <div className="text-center pt-16 pb-14 animate-fade-up">
        <span className="inline-flex items-center gap-1.5 text-[10px] text-blue-400/60 border border-blue-400/20 px-4 py-1 rounded-full uppercase tracking-widest font-bold">
          <Radar size={11} /> Advanced AI Security
        </span>
        <h1 className="text-white text-5xl sm:text-6xl font-bold mt-6 tracking-tight leading-[1.1]">
          Detect Deepfakes<br />Before They Spread
        </h1>
        <p className="text-gray-400 text-base mt-5 max-w-lg mx-auto leading-relaxed">
          Upload an image or video and get a forensic breakdown of frequency artifacts,
          noise inconsistency, and edge softness, backed by a trainable CNN pipeline.
        </p>
        <button
          onClick={onStartScan}
          className="mt-8 inline-flex items-center gap-2 px-10 py-4 rounded-2xl bg-blue-600 text-white font-bold hover:bg-blue-500 transition-all shadow-lg shadow-blue-900/40"
        >
          Scan a File <ArrowRight size={16} />
        </button>
      </div>

      {/* Feature grid */}
      <div className="grid sm:grid-cols-2 gap-4 mb-16 animate-fade-up">
        {FEATURES.map(({ icon: Icon, title, desc }) => (
          <div key={title} className="bg-[#111a2e]/60 border border-blue-500/10 rounded-3xl p-6 hover:border-blue-500/25 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-4">
              <Icon size={18} className="text-blue-400" />
            </div>
            <h3 className="text-white font-semibold mb-1.5">{title}</h3>
            <p className="text-gray-500 text-sm leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>

      {/* Pipeline strip */}
      <div className="bg-[#111a2e]/40 border border-blue-500/10 rounded-3xl p-8 animate-fade-up">
        <h2 className="text-white font-semibold text-lg mb-1 text-center">Three-Tier Detection Pipeline</h2>
        <p className="text-gray-500 text-sm text-center mb-8">The system always picks the strongest model that's actually trained</p>
        <div className="grid sm:grid-cols-3 gap-6">
          {PIPELINE.map(({ step, title, desc }, i) => (
            <div key={step} className="relative text-center">
              <span className="text-4xl font-bold text-blue-500/15">{step}</span>
              <h4 className="text-white font-semibold mt-1">{title}</h4>
              <p className="text-gray-500 text-xs mt-1.5 leading-relaxed">{desc}</p>
              {i < PIPELINE.length - 1 && (
                <ArrowRight size={16} className="hidden sm:block text-blue-500/20 absolute top-6 -right-3" />
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center justify-center gap-2 text-gray-600 text-xs mt-10">
        <ShieldCheck size={13} /> Results are a forensic signal, not a legal or journalistic determination
      </div>
    </div>
  );
}