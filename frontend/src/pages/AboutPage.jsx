import React from 'react';
import { Waves, Volume2, Contrast, Palette, ArrowRight } from 'lucide-react';

const SIGNALS = [
  { icon: Waves, title: 'Frequency Artifacts', desc: 'GAN and diffusion decoders leave characteristic spectral checkerboard patterns. Measured via the ratio of high-frequency to total FFT energy.' },
  { icon: Volume2, title: 'Noise Consistency', desc: 'Real camera sensors produce fairly consistent noise texture; generative models often over-smooth skin. Measured via local Laplacian variance.' },
  { icon: Contrast, title: 'Edge Sharpness', desc: 'Synthesis and blending seams often soften edges around the face boundary. Measured via Canny edge strength.' },
  { icon: Palette, title: 'Color Consistency', desc: 'Face-swap blending frequently leaves chroma inconsistency at the swap boundary, measured in YCrCb space.' },
];

export default function AboutPage({ onStartScan }) {
  return (
    <div className="w-full max-w-3xl mx-auto px-6 pt-10 pb-24">
      <div className="animate-fade-up mb-12">
        <h1 className="text-white text-3xl font-bold tracking-tight mb-3">How It Works</h1>
        <p className="text-gray-400 text-sm leading-relaxed">
          This system runs a three-tier detection pipeline and always uses the strongest model
          that's actually available, falling back gracefully when nothing has been trained yet.
        </p>
      </div>

      <div className="flex flex-col gap-4 mb-12 animate-fade-up">
        <TierCard
          num="1"
          title="Heuristic Forensic Analyzer"
          badge="Always available"
          badgeColor="emerald"
          desc="Four independent signal-processing measurements, combined into a weighted score. No training data required, but it's not a validated classifier. Treat the percentage as a signal, not a verdict."
        />
        <TierCard
          num="2"
          title="Trained CNN (ResNet18)"
          badge="Requires training"
          badgeColor="amber"
          desc="A convolutional network trained from scratch on a labeled real/fake face dataset. Automatically takes over from the heuristic analyzer once weights/deepfake_cnn.pth exists."
        />
        <TierCard
          num="3"
          title="Trained Xception / InceptionV3+GRU"
          badge="Best accuracy"
          badgeColor="blue"
          desc="Transfer-learning models fine-tuned on the 140k Real and Fake Faces dataset. Reaches the highest measured accuracy of the three tiers once trained and evaluated."
        />
      </div>

      <div className="animate-fade-up mb-12">
        <h2 className="text-white text-xl font-bold mb-1">The Four Forensic Signals</h2>
        <p className="text-gray-500 text-sm mb-6">What the heuristic analyzer actually measures</p>
        <div className="grid sm:grid-cols-2 gap-4">
          {SIGNALS.map(({ icon: Icon, title, desc }) => (
            <div key={title} className="bg-[#111a2e]/60 border border-blue-500/10 rounded-2xl p-5">
              <div className="flex items-center gap-2 mb-2">
                <Icon size={16} className="text-blue-400" />
                <h3 className="text-white font-semibold text-sm">{title}</h3>
              </div>
              <p className="text-gray-500 text-xs leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="text-center animate-fade-up">
        <button
          onClick={onStartScan}
          className="inline-flex items-center gap-2 px-10 py-4 rounded-2xl bg-blue-600 text-white font-bold hover:bg-blue-500 transition-all"
        >
          Try a Scan <ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}

function TierCard({ num, title, badge, badgeColor, desc }) {
  const colors = {
    emerald: 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300',
    amber: 'bg-amber-500/10 border-amber-500/20 text-amber-300',
    blue: 'bg-blue-500/10 border-blue-500/20 text-blue-300',
  };
  return (
    <div className="flex gap-4 bg-[#111a2e]/60 border border-blue-500/10 rounded-2xl p-5">
      <div className="w-8 h-8 rounded-full bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 font-bold text-sm shrink-0">
        {num}
      </div>
      <div>
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <h3 className="text-white font-semibold text-sm">{title}</h3>
          <span className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${colors[badgeColor]}`}>{badge}</span>
        </div>
        <p className="text-gray-500 text-xs leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}