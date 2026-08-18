import React from 'react';
import { Home, ScanFace, History, Info, Menu, X } from 'lucide-react';
import Logo from './Logo';

const NAV_ITEMS = [
  { id: 'home', label: 'Home', icon: Home },
  { id: 'scan', label: 'Scan', icon: ScanFace },
  { id: 'history', label: 'History', icon: History },
  { id: 'about', label: 'How It Works', icon: Info },
];

const MODE_LABEL = {
  'heuristic-forensic-analyzer': 'Heuristic Mode',
  'trained-torch-cnn': 'Trained CNN Active',
  'trained-keras': 'Trained Model Active',
};

export default function Navbar({ page, setPage, backendStatus }) {
  const [mobileOpen, setMobileOpen] = React.useState(false);

  const statusDotColor = backendStatus === null
    ? 'bg-amber-400 animate-pulse'
    : backendStatus === 'offline'
      ? 'bg-red-500'
      : 'bg-emerald-400';

  const statusText = backendStatus === null
    ? 'Connecting…'
    : backendStatus === 'offline'
      ? 'Backend offline'
      : (MODE_LABEL[backendStatus.mode] || backendStatus.mode);

  return (
    <nav className="w-full sticky top-0 z-30 bg-[#0b1121]/85 backdrop-blur-md border-b border-white/5">
      <div className="max-w-5xl mx-auto flex items-center justify-between px-6 py-3.5">
        <button onClick={() => setPage('home')} className="flex items-center gap-2 shrink-0">
          <Logo size={22} />
        </button>

        <div className="hidden md:flex items-center gap-1">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
                page === id
                  ? 'bg-blue-500/10 text-blue-300 border border-blue-500/20'
                  : 'text-gray-400 hover:text-white hover:bg-white/5 border border-transparent'
              }`}
            >
              <Icon size={14} /> {label}
            </button>
          ))}
        </div>

        <div className="hidden md:flex items-center gap-1.5 text-[11px] shrink-0">
          <span className={`w-1.5 h-1.5 rounded-full ${statusDotColor}`} />
          <span className="text-gray-500">{statusText}</span>
        </div>

        <button className="md:hidden text-gray-400" onClick={() => setMobileOpen(!mobileOpen)}>
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {mobileOpen && (
        <div className="md:hidden flex flex-col gap-1 px-6 pb-4">
          {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => { setPage(id); setMobileOpen(false); }}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium text-left ${
                page === id ? 'bg-blue-500/10 text-blue-300' : 'text-gray-400'
              }`}
            >
              <Icon size={14} /> {label}
            </button>
          ))}
          <div className="flex items-center gap-1.5 text-[11px] px-4 pt-2">
            <span className={`w-1.5 h-1.5 rounded-full ${statusDotColor}`} />
            <span className="text-gray-500">{statusText}</span>
          </div>
        </div>
      )}
    </nav>
  );
}