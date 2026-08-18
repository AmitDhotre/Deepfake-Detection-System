<p align="center">
  <img src="./frontend/src/assets/veritas-icon.svg" alt="Veritas logo" width="70" />
</p>

<h1 align="center">🛡️ VERITAS</h1>
<p align="center"><b>Intelligent Deepfake Detection Platform</b></p>

## 📌 Project Overview

A full-stack AI-powered **Deepfake Detection Platform** that analyzes images and videos to determine whether they're authentic or AI-generated/manipulated. Veritas runs a three-tier detection pipeline — a heuristic forensic-signal analyzer, a PyTorch CNN, and a Keras Xception/InceptionV3+GRU model — and automatically uses the strongest model actually available, with a multi-page interactive React dashboard for results.

---

## 🚀 Features

- 🎥 Image & Video Deepfake Detection
- 📊 Forensic Signal Breakdown (frequency artifacts, noise consistency, edge sharpness, chroma consistency)
- 🧠 Auto-Selecting 3-Tier Pipeline (heuristic → trained CNN → trained Xception)
- 📈 Confidence Gauge, Radar Chart & Per-Frame Timeline for video
- 🕓 Local Scan History with expandable signal detail
- 🌐 Multi-Page Interactive Dashboard (Home, Scan, History, How It Works)
- ⚡ FastAPI Backend with Automatic Model Detection
- 🏋️ Trainable on a Real 140k-Image Labeled Dataset

---

## 🛠️ Technologies Used

### Backend
- Python
- FastAPI
- Uvicorn
- OpenCV
- NumPy
- Pillow

### Models
- PyTorch (ResNet18)
- TensorFlow / Keras
- Xception
- InceptionV3 + GRU

### Frontend
- React
- Vite
- Tailwind CSS
- Recharts
- Lucide Icons

### Dataset
- Kaggle "140k Real and Fake Faces" (FFHQ real vs. GAN-generated fake)

---

## 📂 Project Structure

```text
veritas/
├── backend/
│   ├── main.py                  FastAPI app, auto-selects strongest model
│   ├── detector.py              Heuristic forensic analyzer
│   ├── cnn_inference.py         Torch CNN inference
│   ├── keras_inference.py       Keras Xception/GRU inference
│   ├── train.py / train_image.py / train_video.py / train_image_csv.py
│   ├── evaluate.py              Precision/recall/F1/ROC-AUC on held-out test set
│   ├── download_dataset.py      kagglehub dataset fetcher
│   ├── dataset_meta/            train.csv / valid.csv / test.csv
│   ├── weights/                 Trained model files land here
│   └── requirements.txt / requirements-train.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── assets/
│       │   └── veritas-icon.svg     Logo mark
│       ├── pages/                   Home, Scan, History, About
│       └── components/              Navbar, Logo, Footer, UploadView, ResultView, SignalBars, ScanHistory
└── README.md
```

---

## ⚙️ Working Process

### 1️⃣ Media Upload
User drags and drops an image or video into the dashboard.

### 2️⃣ Face Detection
Haar cascade localizes and crops the face region.

### 3️⃣ Signal Extraction / Inference
Forensic signals are computed, or CNN/Xception inference is executed.

### 4️⃣ Model Auto-Selection
Backend picks the strongest trained model available, falling back to the heuristic analyzer if none is trained yet.

### 5️⃣ Score Aggregation
Combined confidence score computed using per-frame scoring and temporal variance for video.

### 6️⃣ Result Delivery
JSON response is returned to the frontend and rendered as a confidence gauge, signal radar chart, forensic signal bars, and frame timeline.

---

## 📊 ML/CV Concepts Used

- Convolutional Neural Networks (ResNet18, Xception)
- Transfer Learning (ImageNet-pretrained backbones)
- Recurrent Neural Networks (GRU) for temporal video analysis
- Digital Image Forensics
  - FFT frequency analysis
  - Noise variance analysis
  - Edge detection
  - Color-space analysis
- Binary Classification

---

## 📸 Application Preview

### Home Page
Hero section, feature highlights, and 3-tier pipeline explainer.

### Scan Page
Drag-and-drop upload, live analysis, confidence gauge, forensic signal breakdown, and frame-by-frame timeline.

### History Page
Past scans with expandable per-scan signal detail.

### About Page
Explains each forensic signal and every pipeline tier in plain language.

---

## ▶️ Run Locally

### Backend

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

---

## 🏋️ Train on the Real Dataset

```powershell
pip install -r requirements-train.txt

python train_image_csv.py --images_root "D:\path\to\real-vs-fake" --out weights/xception_deepfake_image.h5

python evaluate.py --images_root "D:\path\to\real-vs-fake"
```

---

## 🎯 Future Enhancements

- Grad-CAM heatmap showing *where* the model detected manipulation
- Audio deepfake detection for video soundtracks
- Batch scanning (multiple files at once)
- API key authentication and rate limiting
- Docker containerization and cloud hosting
- Face-swap boundary localization overlay

---

## 👨‍💻 Author

**Amit Dhotre**

Computer Engineering Student | AI & Data Science Enthusiast

---

## ⭐ Support

If you found Veritas useful, consider giving it a star on GitHub.
