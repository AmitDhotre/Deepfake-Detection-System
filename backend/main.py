from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

# NEW: Hugging Face model downloader
from huggingface_hub import hf_hub_download

import detector

app = FastAPI(title="Deepfake Detection API")

# In production, set ALLOWED_ORIGINS to your deployed frontend's URL
# (comma-separated for multiple), e.g.:
#   ALLOWED_ORIGINS=https://your-app.vercel.app
# Falls back to "*" (allow everything) for local development so you don't
# have to configure anything to run this on your machine.
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins.split(",")] if _allowed_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Create weights directory if missing
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ---------------------------------------------------
# DOWNLOAD MODEL FROM HUGGING FACE IF NOT PRESENT
# ---------------------------------------------------
KERAS_IMAGE_PATH = os.path.join(WEIGHTS_DIR, "xception_deepfake_image.h5")

if not os.path.exists(KERAS_IMAGE_PATH):
    try:
        print("Downloading model from Hugging Face...")

        hf_hub_download(
            repo_id="amitdhotre/veritas-model",
            filename="xception_deepfake_image.h5",
            local_dir=WEIGHTS_DIR
        )

        print("Model downloaded successfully.")

    except Exception as e:
        print(f"Model download failed: {e}")

# ---------------------------------------------------

KERAS_VIDEO_PATH = os.path.join(WEIGHTS_DIR, "video_gru.keras")
TORCH_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "deepfake_cnn.pth")

KERAS_IMAGE_AVAILABLE = os.path.exists(KERAS_IMAGE_PATH)
KERAS_VIDEO_AVAILABLE = os.path.exists(KERAS_VIDEO_PATH)
TORCH_AVAILABLE = os.path.exists(TORCH_WEIGHTS_PATH)

if KERAS_IMAGE_AVAILABLE or KERAS_VIDEO_AVAILABLE:
    import keras_inference

if TORCH_AVAILABLE:
    from cnn_inference import score_with_cnn


@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()
    ext = os.path.splitext(filename)[1]
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file")

    is_video = ext in VIDEO_EXTS or (file.content_type or "").startswith("video")
    is_image = ext in IMAGE_EXTS or (file.content_type or "").startswith("image")

    if not (is_video or is_image):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        if is_image and KERAS_IMAGE_AVAILABLE:
            analysis = keras_inference.score_image(content)
            model_name = "Xception (transfer learning, notebook-trained)"

        elif is_video and KERAS_VIDEO_AVAILABLE:
            analysis = keras_inference.score_video(content)
            model_name = "InceptionV3+GRU (notebook-trained)"

        elif TORCH_AVAILABLE:
            analysis = score_with_cnn(content, is_video)
            model_name = detector.MODEL_NAME_CNN

        elif is_video:
            analysis = detector.analyze_video_bytes(content)
            model_name = detector.MODEL_NAME_HEURISTIC

        else:
            analysis = detector.analyze_image_bytes(content)
            model_name = detector.MODEL_NAME_HEURISTIC

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    combined = analysis["combined_score"]
    is_deepfake = combined >= 0.5
    confidence_pct = round((combined if is_deepfake else (1 - combined)) * 100, 1)

    return {
        "threat": "DEEPFAKE" if is_deepfake else "NOT DEEPFAKE",
        "percentage": f"{confidence_pct}%",
        "raw_score": round(combined, 3),
        "model": model_name,
        "media_type": "video" if is_video else "image",
        "frame_count": analysis["frame_count"],
        "faces_detected": analysis["faces_detected"],
        "per_frame_scores": analysis["per_frame_scores"],
        "signals": analysis["signals"],
        "temporal_variance": analysis["temporal_variance"],
    }


@app.get("/")
async def root():
    if KERAS_IMAGE_AVAILABLE or KERAS_VIDEO_AVAILABLE:
        mode = "trained-keras"
    elif TORCH_AVAILABLE:
        mode = "trained-torch-cnn"
    else:
        mode = "heuristic-forensic-analyzer"

    return {
        "status": "Backend is running successfully",
        "mode": mode,
        "keras_image_model": KERAS_IMAGE_AVAILABLE,
        "kesras_video_model": KERAS_VIDEO_AVAILABLE,
        "torch_model": TORCH_AVAILABLE,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)