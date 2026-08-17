import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { ShieldAlert, ShieldCheck, RotateCcw, ScanFace, Layers, Activity, Info } from 'lucide-react';
import SignalBars from './SignalBars';

function GaugeRing({ percent, isFake }) {
  const color = isFake ? '#ef4444' : '#22c55e';
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (percent / 100) * circumference;
  return (
    <div className="relative w-36 h-36 shrink-0">
      <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
        <circle cx="60" cy="60" r="54" fill="none" stroke="#1c273f" strokeWidth="10" />
        <circle
          cx="60" cy="60" r="54" fill="none" stroke={color} strokeWidth="10"
          strokeDasharray={circumference} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.4,0,0.2,1)' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-white tabular-nums">{percent}%</span>
        <span className="text-[9px] uppercase tracking-widest text-gray-500 mt-1">confidence</span>
      </div>
    </div>
  );
}

function MediaPreview({ mediaUrl, mediaType, isFake }) {
  if (!mediaUrl) return null;
  const ringColor = isFake ? 'border-red-500/40' : 'border-emerald-500/40';
  const cornerColor = isFake ? '#ef4444' : '#22c55e';
  return (
    <div className={`relative w-full max-w-xs mx-auto rounded-2xl overflow-hidden border ${ringColor} bg-black/50 animate-fade-up`}>
      {mediaType === 'video' ? (
        <video src={mediaUrl} className="w-full max-h-64 object-contain" controls />
      ) : (
        <img src={mediaUrl} alt="scanned media" className="w-full max-h-64 object-contain" />
      )}
      {['top-2 left-2', 'top-2 right-2 rotate-90', 'bottom-2 right-2 rotate-180', 'bottom-2 left-2 -rotate-90'].map((pos, i) => (
        <svg key={i} className={`absolute ${pos} w-5 h-5 pointer-events-none`} viewBox="0 0 20 20">
          <path d="M0 0 L0 8 M0 0 L8 0" stroke={cornerColor} strokeWidth="2" fill="none" />
        </svg>
      ))}
    </div>
  );
}

export default function ResultView({ result, onReset, mediaUrl }) {
  const isFake = result.threat === 'DEEPFAKE';
  const percent = parseFloat(result.percentage);
  const isHeuristic = (result.model || '').toLowerCase().includes('heuristic');

  const frameData = (result.per_frame_scores || []).map((score, i) => ({
    frame: i + 1,
    score: Math.round(score * 100),
  }));

  return (
    <div className="w-full flex flex-col gap-6">
      <div className={`rounded-3xl p-6 flex flex-col sm:flex-row items-center gap-6 border animate-fade-up ${
        isFake ? 'bg-red-500/[0.06] border-red-500/20' : 'bg-emerald-500/[0.06] border-emerald-500/20'
      }`}>
        <GaugeRing percent={percent} isFake={isFake} />
        <div className="flex-1 min-w-0 text-center sm:text-left">
          <div className="flex items-center justify-center sm:justify-start gap-2">
            {isFake ? <ShieldAlert className="text-red-400" size={22} /> : <ShieldCheck className="text-emerald-400" size={22} />}
            <h2 className={`text-2xl font-bold tracking-tight ${isFake ? 'text-red-400 threat-glow' : 'text-emerald-400'}`}>
              {isFake ? 'Likely Deepfake' : 'Likely Authentic'}
            </h2>
          </div>
          <p className="text-gray-400 text-sm mt-2 leading-relaxed">
            {isFake
              ? "This media shows patterns consistent with AI generation or manipulation."
              : "This media shows patterns consistent with an unmanipulated capture."}
          </p>
          <div className="flex flex-wrap justify-center sm:justify-start gap-2 mt-3">
            <span className="text-[11px] bg-blue-500/10 border border-blue-500/20 text-blue-300 px-3 py-1 rounded-full font-medium">
              {result.model}
            </span>
            <span className="text-[11px] bg-slate-700/40 border border-slate-600/30 text-gray-300 px-3 py-1 rounded-full font-medium capitalize">
              {result.media_type}
            </span>
          </div>
        </div>
      </div>

      {mediaUrl && <MediaPreview mediaUrl={mediaUrl} mediaType={result.media_type} isFake={isFake} />}

      {isHeuristic && (
        <div className="flex items-start gap-2 bg-amber-500/[0.06] border border-amber-500/20 text-amber-300 text-xs rounded-2xl p-4 animate-fade-up">
          <Info size={16} className="shrink-0 mt-0.5" />
          <p>
            Running in <strong>heuristic mode</strong> — no trained model is loaded. This score comes from
            forensic signal-processing rules, not a validated classifier. Train a model for measured accuracy.
          </p>
        </div>
      )}

      <div className="grid grid-cols-3 gap-4 animate-fade-up">
        <StatCard icon={<Layers size={16} />} label="Frames Analyzed" value={result.frame_count ?? '—'} />
        <StatCard icon={<ScanFace size={16} />} label="Faces Detected" value={result.faces_detected ?? '—'} />
        <StatCard icon={<Activity size={16} />} label="Temporal Variance" value={result.temporal_variance ?? 'n/a'} />
      </div>

      {Object.keys(result.signals || {}).length > 1 && (
        <div className="bg-[#111a2e]/60 border border-blue-500/10 rounded-3xl p-6 animate-fade-up">
          <h3 className="text-white font-semibold mb-1">Forensic Signal Breakdown</h3>
          <p className="text-gray-500 text-xs mb-4">Higher = more suspicious for that signal</p>
          <SignalBars signals={result.signals} />
        </div>
      )}

      {frameData.length > 1 && (
        <div className="bg-[#111a2e]/60 border border-blue-500/10 rounded-3xl p-6 animate-fade-up">
          <h3 className="text-white font-semibold mb-1">Frame-by-Frame Score</h3>
          <p className="text-gray-500 text-xs mb-4">Fake-likelihood across sampled frames</p>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={frameData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1c273f" />
              <XAxis dataKey="frame" tick={{ fill: '#64748b', fontSize: 11 }} label={{ value: 'Frame', position: 'insideBottom', offset: -3, fill: '#64748b', fontSize: 11 }} />
              <YAxis domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 11 }} />
              <Tooltip contentStyle={{ background: '#0b1121', border: '1px solid #1c273f', borderRadius: 12, fontSize: 12 }} />
              <Line type="monotone" dataKey="score" stroke="#4da3ff" strokeWidth={2} dot={{ r: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <button
        onClick={onReset}
        className="self-center flex items-center gap-2 px-10 py-3.5 rounded-2xl bg-slate-800/50 text-gray-300 font-bold hover:bg-slate-800 hover:text-white transition-all"
      >
        <RotateCcw size={16} /> Scan Another File
      </button>
    </div>
  );
}

function StatCard({ icon, label, value }) {
  return (
    <div className="bg-[#111a2e]/60 border border-blue-500/10 rounded-2xl p-4 flex flex-col gap-1">
      <div className="flex items-center gap-1.5 text-blue-400/70">{icon}<span className="text-[10px] uppercase tracking-widest text-gray-500">{label}</span></div>
      <span className="text-white text-xl font-bold">{String(value)}</span>
    </div>
  );
}
