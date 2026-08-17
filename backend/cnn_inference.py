"""
Loaded by main.py only when backend/weights/deepfake_cnn.pth exists
(i.e. after you've run train.py on a real dataset). Mirrors the
resnet18 architecture defined in train.py.
"""

import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models

import detector

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "deepfake_cnn.pth")

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None


def _load_model():
    global _model
    if _model is None:
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, 2)
        m.load_state_dict(torch.load(_WEIGHTS_PATH, map_location=_DEVICE))
        m.eval().to(_DEVICE)
        _model = m
    return _model


def _score_bgr_frame(bgr_face):
    rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
    tensor = _TRANSFORM(rgb).unsqueeze(0).to(_DEVICE)
    model = _load_model()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[0])  # index 0 = 'fake' (ImageFolder alphabetical: fake, real)


def score_with_cnn(content_bytes, is_video):
    if not is_video:
        arr = np.frombuffer(content_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image file")
        crop, n_faces = detector._detect_largest_face(img)
        if crop is None or crop.size == 0:
            crop = img
        score = _score_bgr_frame(crop)
        return {
            "frame_count": 1,
            "faces_detected": n_faces,
            "per_frame_scores": [round(score, 3)],
            "signals": {"cnn_confidence": round(score, 3)},
            "combined_score": score,
            "temporal_variance": None,
        }

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(int(fps / 2), 1)
        scores, faces_total, frames_analyzed, idx = [], 0, 0, 0
        while frames_analyzed < 40:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                crop, n_faces = detector._detect_largest_face(frame)
                if crop is None or crop.size == 0:
                    crop = frame
                else:
                    faces_total += 1
                scores.append(round(_score_bgr_frame(crop), 3))
                frames_analyzed += 1
            idx += 1
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not scores:
        raise ValueError("Could not extract any frames from this video")

    combined = sum(scores) / len(scores)
    return {
        "frame_count": frames_analyzed,
        "faces_detected": faces_total,
        "per_frame_scores": scores,
        "signals": {"cnn_confidence": round(combined, 3)},
        "combined_score": combined,
        "temporal_variance": round(float(np.var(scores)), 4) if len(scores) > 1 else 0.0,
    }
