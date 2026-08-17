import React from 'react';
import { History, Trash2, ShieldAlert, ShieldCheck } from 'lucide-react';

const STORAGE_KEY = 'deepfake_scan_history';
const MAX_ENTRIES = 12;

export function loadScanHistory() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveScanToHistory(entry) {
  const existing = loadScanHistory();
  const updated = [
    { ...entry, scannedAt: new Date().toISOString() },
    ...existing,
  ].slice(0, MAX_ENTRIES);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
  return updated;
}

export function clearScanHistory() {
  localStorage.removeItem(STORAGE_KEY);
}

export default function ScanHistory({ history, onClear }) {
  if (!history || history.length === 0) return null;

  return (
    <div className="w-full max-w-3xl mt-10">
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center gap-2 text-gray-400">
          <History size={15} />
          <span className="text-xs uppercase tracking-widest font-semibold">Recent Scans</span>
        </div>
        <button
          onClick={onClear}
          className="flex items-center gap-1 text-[11px] text-gray-500 hover:text-red-400 transition-colors"
        >
          <Trash2 size={12} /> Clear
        </button>
      </div>
      <div className="flex flex-col gap-2">
        {history.map((item, i) => {
          const isFake = item.threat === 'DEEPFAKE';
          return (
            <div
              key={i}
              className="flex items-center justify-between bg-[#111a2e]/50 border border-blue-500/10 rounded-2xl px-4 py-3"
            >
              <div className="flex items-center gap-3 min-w-0">
                {isFake ? (
                  <ShieldAlert size={16} className="text-red-400 shrink-0" />
                ) : (
                  <ShieldCheck size={16} className="text-emerald-400 shrink-0" />
                )}
                <span className="text-gray-300 text-sm truncate max-w-[220px]">{item.filename}</span>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                <span className={`text-xs font-bold ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
                  {item.percentage}
                </span>
                <span className="text-[10px] text-gray-600">
                  {item.scannedAt ? new Date(item.scannedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
