import React, { useRef, useState } from 'react';
import { UploadCloud, FileVideo, FileImage, X } from 'lucide-react';

const ACCEPTED = '.jpg,.jpeg,.png,.webp,.bmp,.mp4,.mov,.avi,.mkv,.webm';

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

export default function UploadView({ onUpload, file }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const isVideo = file && (file.type.startsWith('video') || /\.(mp4|mov|avi|mkv|webm)$/i.test(file.name));
  const previewUrl = file ? URL.createObjectURL(file) : null;

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onUpload({ target: { files: e.dataTransfer.files } });
    }
  };

  const clearFile = (e) => {
    e.stopPropagation();
    onUpload({ target: { files: null } });
  };

  if (file) {
    return (
      <div className="w-full flex flex-col items-center gap-6">
        <div className="relative w-full max-w-md rounded-3xl overflow-hidden border border-blue-500/20 bg-black/40">
          {isVideo ? (
            <video src={previewUrl} className="w-full max-h-80 object-contain bg-black" controls />
          ) : (
            <img src={previewUrl} alt="preview" className="w-full max-h-80 object-contain bg-black" />
          )}
          <button
            onClick={clearFile}
            className="absolute top-3 right-3 bg-black/60 hover:bg-black/80 text-white rounded-full p-1.5 transition-colors"
            aria-label="Remove file"
          >
            <X size={16} />
          </button>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          {isVideo ? <FileVideo size={16} className="text-blue-400" /> : <FileImage size={16} className="text-blue-400" />}
          <span className="text-gray-200 font-medium">{file.name}</span>
          <span className="text-gray-500">· {formatBytes(file.size)}</span>
        </div>
      </div>
    );
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      className={`w-full flex flex-col items-center justify-center gap-4 py-16 cursor-pointer rounded-[36px] transition-all border-2 border-dashed ${
        dragging ? 'border-blue-400 bg-blue-500/5' : 'border-transparent hover:bg-white/[0.02]'
      }`}
    >
      <div className="w-16 h-16 rounded-2xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
        <UploadCloud className="text-blue-400" size={28} />
      </div>
      <div className="text-center">
        <p className="text-white font-semibold text-lg">Drop an image or video to scan</p>
        <p className="text-gray-500 text-sm mt-1">or click to browse · JPG, PNG, MP4, MOV, WEBM</p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        onChange={onUpload}
        className="hidden"
      />
    </div>
  );
}
