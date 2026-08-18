import React from 'react';

// Shield icon inline so it always renders correctly regardless of how
// the build tool handles raw .svg imports.
function VeritasIcon({ size = 28 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 160 210" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Veritas logo mark">
      <path d="M80,10 L150,40 L150,95 C150,145 120,180 80,200 C40,180 10,145 10,95 L10,40 Z" fill="none" stroke="#4da3ff" strokeWidth="4"/>
      <line x1="10" y1="110" x2="150" y2="110" stroke="#4da3ff" strokeWidth="2" strokeDasharray="4 3"/>
      <circle cx="80" cy="75" r="26" fill="none" stroke="#4da3ff" strokeWidth="3"/>
      <path d="M56,95 C56,130 104,130 104,95" fill="none" stroke="#4da3ff" strokeWidth="3"/>
      <rect x="92" y="120" width="10" height="10" fill="#4da3ff" opacity="0.8"/>
      <rect x="108" y="132" width="7" height="7" fill="#4da3ff" opacity="0.6"/>
      <rect x="86" y="140" width="14" height="6" fill="#4da3ff" opacity="0.5"/>
      <rect x="112" y="118" width="6" height="14" fill="#4da3ff" opacity="0.7"/>
    </svg>
  );
}

export default function Logo({ size = 28, showTagline = false, className = '' }) {
  return (
    <div className={`flex flex-col items-center gap-1 ${className}`}>
      <div className="flex items-center gap-2">
        <VeritasIcon size={size} />
        <span className="text-lg font-semibold tracking-wide text-gray-200">
          VERIT<span className="text-blue-400">AS</span>
        </span>
      </div>
      {showTagline && (
        <span className="text-[11px] text-gray-500 text-center">
          Intelligent Deepfake Detection Platform
        </span>
      )}
    </div>
  );
}