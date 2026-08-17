import React from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip,
} from 'recharts';
import { Waves, Volume2, Contrast, Palette, Cpu } from 'lucide-react';

export const SIGNAL_META = {
  frequency_artifact: { label: 'Frequency Artifacts', icon: Waves, hint: 'GAN upsampling checkerboard patterns' },
  noise_consistency: { label: 'Noise Inconsistency', icon: Volume2, hint: 'Over-smoothed sensor-noise texture' },
  edge_sharpness: { label: 'Edge Softness', icon: Contrast, hint: 'Blending seams around face boundary' },
  color_consistency: { label: 'Chroma Inconsistency', icon: Palette, hint: 'Color mismatch at swap boundary' },
  cnn_confidence: { label: 'Model Confidence', icon: Cpu, hint: 'Trained classifier output' },
};

export default function SignalBars({ signals, showRadar = true, compact = false }) {
  const entries = Object.entries(signals || {});
  if (entries.length <= 1) return null;

  const radarData = entries.map(([key, value]) => ({
    signal: (SIGNAL_META[key] || {}).label || key,
    value: Math.round(value * 100),
  }));

  return (
    <div>
      <div className="flex flex-col gap-4">
        {entries.map(([key, value]) => {
          const meta = SIGNAL_META[key] || { label: key, icon: Cpu, hint: '' };
          const Icon = meta.icon;
          const pct = Math.round(value * 100);
          const barColor = pct >= 60 ? '#ef4444' : pct >= 35 ? '#f59e0b' : '#22c55e';
          return (
            <div key={key}>
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2 text-gray-300 text-sm">
                  <Icon size={14} className="text-blue-400/70" />
                  <span>{meta.label}</span>
                </div>
                <span className="text-xs font-semibold text-gray-400 tabular-nums">{pct}%</span>
              </div>
              <div className="h-2 w-full bg-slate-800/60 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{ width: `${pct}%`, backgroundColor: barColor }}
                />
              </div>
              {!compact && meta.hint && <p className="text-[11px] text-gray-600 mt-1">{meta.hint}</p>}
            </div>
          );
        })}
      </div>

      {showRadar && (
        <div className="mt-6 pt-6 border-t border-blue-500/10">
          <ResponsiveContainer width="100%" height={compact ? 180 : 220}>
            <RadarChart data={radarData} outerRadius="75%">
              <PolarGrid stroke="#1c273f" />
              <PolarAngleAxis dataKey="signal" tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Radar dataKey="value" stroke="#4da3ff" fill="#4da3ff" fillOpacity={0.35} />
              <Tooltip contentStyle={{ background: '#0b1121', border: '1px solid #1c273f', borderRadius: 12, fontSize: 12 }} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
