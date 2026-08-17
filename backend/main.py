from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

import detector

app = FastAPI(title="Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

# Model priority, highest first. main.py picks the strongest available
# option automatically — no code changes needed as you add trained weights.
#   1. Keras models trained via train_image.py / train_video.py
#      (Xception transfer learning / InceptionV3+GRU, ported from the
#      reference notebook — the strongest option once trained)
#   2. Torch CNN trained via train.py (ResNet18 from scratch)
#   3. Heuristic forensic-signal analyzer in detector.py (no training required,
#      but not a validated classifier — see README)
KERAS_IMAGE_PATH = os.path.join(WEIGHTS_DIR, "xception_deepfake_image.h5")
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
        "keras_video_model": KERAS_VIDEO_AVAILABLE,
        "torch_model": TORCH_AVAILABLE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
