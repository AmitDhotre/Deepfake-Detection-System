import React, { useState, useEffect, useRef } from 'react';
import './index.css';
import Navbar from './components/Navbar';
import HomePage from './pages/HomePage';
import ScanPage from './pages/ScanPage';
import HistoryPage from './pages/HistoryPage';
import AboutPage from './pages/AboutPage';
import { saveScanToHistory, loadScanHistory, clearScanHistory } from './components/ScanHistory';

const API_BASE = 'http://localhost:8000';

function App() {
  const [page, setPage] = useState('home');
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [backendStatus, setBackendStatus] = useState(null);
  const prevUrlRef = useRef(null);

  useEffect(() => {
    setHistory(loadScanHistory());
    fetch(`${API_BASE}/`).then(r => r.json()).then(setBackendStatus).catch(() => setBackendStatus('offline'));
  }, []);

  const goToScan = () => setPage('scan');

  const handleFileChange = (e) => {
    if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    if (e.target.files && e.target.files[0]) {
      const f = e.target.files[0];
      const url = URL.createObjectURL(f);
      prevUrlRef.current = url;
      setFile(f);
      setPreviewUrl(url);
      setResult(null);
    } else {
      prevUrlRef.current = null;
      setFile(null);
      setPreviewUrl(null);
    }
  };

  const triggerAnalysis = async () => {
    if (!file) return alert('Select a file first!');
    setAnalyzing(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: formData });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || 'Analysis failed');
      }
      const data = await response.json();
      setResult(data);
      setHistory(saveScanToHistory({ ...data, filename: file.name }));
    } catch (error) {
      alert(error.message || 'Backend Connection Failed! Make sure `python main.py` is running on port 8000.');
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    if (prevUrlRef.current) URL.revokeObjectURL(prevUrlRef.current);
    prevUrlRef.current = null;
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setAnalyzing(false);
  };

  const handleClearHistory = () => {
    clearScanHistory();
    setHistory([]);
  };

  return (
    <div className="min-h-screen w-full bg-[#0b1121] bg-[radial-gradient(circle_at_50%_0%,rgba(77,163,255,0.08),transparent_55%)] flex flex-col">
      <Navbar page={page} setPage={setPage} backendStatus={backendStatus} />

      <main className="flex-1 flex flex-col items-center">
        {page === 'home' && <HomePage onStartScan={goToScan} />}
        {page === 'scan' && (
          <ScanPage
            file={file}
            result={result}
            analyzing={analyzing}
            previewUrl={previewUrl}
            onFileChange={handleFileChange}
            onAnalyze={triggerAnalysis}
            onReset={handleReset}
          />
        )}
        {page === 'history' && (
          <HistoryPage history={history} onClear={handleClearHistory} onStartScan={goToScan} />
        )}
        {page === 'about' && <AboutPage onStartScan={goToScan} />}
      </main>

      <footer className="text-gray-700 text-[11px] text-center py-6">
        Forensic signals + trainable CNN pipeline · results are not a legal determination
      </footer>
    </div>
  );
}

export default App;
