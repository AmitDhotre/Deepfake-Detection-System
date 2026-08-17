import React from 'react';
import UploadView from '../components/UploadView';
import ResultView from '../components/ResultView';

export default function ScanPage({
  file, result, analyzing, previewUrl,
  onFileChange, onAnalyze, onReset,
}) {
  return (
    <div className="w-full max-w-3xl mx-auto px-6 pt-10 pb-24 flex flex-col items-center">
      {!result && (
        <div className="text-center mb-8 animate-fade-up">
          <h1 className="text-white text-3xl font-bold tracking-tight">Scan a File</h1>
          <p className="text-gray-500 text-sm mt-2">Drop an image or video below to run it through the detection pipeline.</p>
        </div>
      )}

      <div className="bg-[#111a2e]/60 border-2 border-dashed border-blue-500/10 rounded-[40px] p-8 sm:p-10 w-full flex flex-col items-center relative overflow-hidden shadow-2xl backdrop-blur-sm">
        {analyzing && <div className="animate-scan" />}
        {result ? (
          <ResultView result={result} onReset={onReset} mediaUrl={previewUrl} />
        ) : (
          <UploadView onUpload={onFileChange} file={file} />
        )}
      </div>

      {!result && (
        <div className="flex gap-4 sm:gap-6 mt-10">
          <button onClick={onReset} className="px-8 sm:px-12 py-4 rounded-2xl bg-slate-800/50 text-gray-400 font-bold hover:bg-slate-800 hover:text-white transition-all">
            Cancel
          </button>
          <button
            onClick={onAnalyze}
            disabled={analyzing || !file}
            className={`px-8 sm:px-12 py-4 rounded-2xl font-bold transition-all ${
              analyzing ? 'bg-blue-900 text-blue-400'
                : !file ? 'bg-slate-800 text-gray-600 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-500'
            }`}
          >
            {analyzing ? 'Analyzing...' : 'Analyze File'}
          </button>
        </div>
      )}
    </div>
  );
}
