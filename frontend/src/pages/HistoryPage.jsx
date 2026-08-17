import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, Trash2, ChevronDown, ScanFace as EmptyIcon, ArrowRight } from 'lucide-react';
import SignalBars from '../components/SignalBars';

export default function HistoryPage({ history, onClear, onStartScan }) {
  const [expandedIdx, setExpandedIdx] = useState(null);

  if (!history || history.length === 0) {
    return (
      <div className="w-full max-w-3xl mx-auto px-6 pt-10 pb-24 flex flex-col items-center text-center">
        <div className="w-14 h-14 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center mb-4">
          <EmptyIcon size={22} className="text-blue-400" />
        </div>
        <h1 className="text-white text-2xl font-bold mb-2">No scans yet</h1>
        <p className="text-gray-500 text-sm mb-6 max-w-sm">
          Your scan history is stored locally in this browser. Run a scan and it'll show up here.
        </p>
        <button
          onClick={onStartScan}
          className="inline-flex items-center gap-2 px-8 py-3.5 rounded-2xl bg-blue-600 text-white font-bold hover:bg-blue-500 transition-all"
        >
          Scan a File <ArrowRight size={15} />
        </button>
      </div>
    );
  }

  const fakeCount = history.filter(h => h.threat === 'DEEPFAKE').length;

  return (
    <div className="w-full max-w-3xl mx-auto px-6 pt-10 pb-24">
      <div className="flex items-center justify-between mb-2 animate-fade-up">
        <h1 className="text-white text-3xl font-bold tracking-tight">Scan History</h1>
        <button
          onClick={onClear}
          className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-red-400 transition-colors"
        >
          <Trash2 size={13} /> Clear All
        </button>
      </div>
      <p className="text-gray-500 text-sm mb-8 animate-fade-up">
        {history.length} scan{history.length !== 1 ? 's' : ''} stored in this browser · {fakeCount} flagged as deepfake
      </p>

      <div className="flex flex-col gap-3">
        {history.map((item, i) => {
          const isFake = item.threat === 'DEEPFAKE';
          const isOpen = expandedIdx === i;
          const hasSignals = item.signals && Object.keys(item.signals).length > 1;
          return (
            <div key={i} className="bg-[#111a2e]/60 border border-blue-500/10 rounded-2xl overflow-hidden animate-fade-up">
              <button
                onClick={() => setExpandedIdx(isOpen ? null : i)}
                className="w-full flex items-center justify-between px-5 py-4 text-left"
              >
                <div className="flex items-center gap-3 min-w-0">
                  {isFake ? (
                    <ShieldAlert size={17} className="text-red-400 shrink-0" />
                  ) : (
                    <ShieldCheck size={17} className="text-emerald-400 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className="text-gray-200 text-sm font-medium truncate max-w-[220px] sm:max-w-xs">{item.filename}</p>
                    <p className="text-gray-600 text-[11px]">
                      {item.scannedAt ? new Date(item.scannedAt).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }) : ''}
                      {' · '}{item.model}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-3 shrink-0">
                  <span className={`text-sm font-bold ${isFake ? 'text-red-400' : 'text-emerald-400'}`}>
                    {item.percentage}
                  </span>
                  {hasSignals && (
                    <ChevronDown size={16} className={`text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
                  )}
                </div>
              </button>
              {isOpen && hasSignals && (
                <div className="px-5 pb-5 pt-1 border-t border-blue-500/10">
                  <SignalBars signals={item.signals} showRadar={false} compact />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
