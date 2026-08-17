import React from 'react';
import { Linkedin, Facebook, Instagram, Mail, Twitter, ShieldHalf } from 'lucide-react';

const SOCIAL_LINKS = [
  { icon: Instagram, href: 'https://www.instagram.com/amit_dhotre_06/', label: 'Instagram' },
  { icon: Linkedin, href: 'https://www.linkedin.com/in/amit-dhotre/', label: 'LinkedIn' },
  { icon: Twitter, href: 'https://x.com/Amitvd93', label: 'X (Twitter)' },
  { icon: Facebook, href: 'https://www.facebook.com/profile.php?id=100072180246893', label: 'Facebook' },
  { icon: Mail, href: 'mailto:amitdhoter976@gaiml.com', label: 'Email' },
];

export default function Footer() {
  return (
    <footer className="w-full border-t border-white/5 mt-10">
      <div className="max-w-3xl mx-auto px-6 py-10 flex flex-col items-center gap-6">
        <div className="flex items-center gap-2 text-gray-500">
          <ShieldHalf size={16} className="text-blue-400" />
          <span className="text-xs font-semibold tracking-wide">DEEPFAKE.AI</span>
        </div>

        <div className="flex items-center gap-3">
          {SOCIAL_LINKS.map((item) => (
            <a key={item.label} href={item.href} target="_blank" rel="noopener noreferrer" aria-label={item.label} className="w-9 h-9 rounded-full bg-slate-800/50 border border-white/5 flex items-center justify-center text-gray-400 hover:text-blue-400 hover:border-blue-500/30 hover:bg-slate-800 transition-all">
              <item.icon size={16} />
            </a>
          ))}
        </div>

        <p className="text-gray-600 text-xs text-center max-w-md leading-relaxed">
          This tool combines forensic signal analysis with a trainable CNN pipeline to flag
          possible deepfakes. Results are a signal, not a legal or journalistic determination.
        </p>

        <p className="text-gray-700 text-[11px] text-center">
          Copyright © 2026 Amit Dhotre. All Rights Reserved.
        </p>
      </div>
    </footer>
  );
}