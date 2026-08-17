"""
Multi-signal forensic deepfake analyzer.

Honesty note (read this before you trust the numbers):
This module does NOT run a trained deepfake-classification neural network.
Training one requires a labeled dataset (FaceForensics++, DFDC, Celeb-DF —
none of which are downloadable from this environment) plus GPU time.

Instead, this computes four independent, well-established forensic signals
that GAN/diffusion-generated and face-swapped faces tend to disturb:

  1. frequency_artifact  - GAN upsampling/decoder layers leave characteristic
                            spectral checkerboard patterns. We measure the
                            ratio of high-frequency to total FFT energy.
  2. noise_consistency    - Real camera sensor noise has a fairly consistent
                            texture; generative models often over-smooth skin.
                            We measure local variance via Laplacian.
  3. edge_sharpness       - Synthesis/blending seams often soften edges
                            around the face boundary.
  4. color_consistency    - Face-swap blending frequently leaves chroma
                            (YCrCb) inconsistency at the swap boundary.

  For video, we add:
  5. temporal_variance    - Frame-to-frame flicker in the above signals.
                            Face-swapped video is often less temporally
                            stable frame-to-frame than a real face.

These are combined into a weighted "fake-likelihood" score in [0, 1].
This is a legitimate, real signal-processing pipeline — but it is a
heuristic, not a trained classifier, and should not be presented as
lab-grade accurate. See train.py to plug in a real trained CNN, which
will automatically take over if a model file is present.
"""

import os
import tempfile
import numpy as np
import cv2

_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(_CASCADE_PATH)

MODEL_NAME_HEURISTIC = "ForensicSignal-Analyzer v1 (heuristic, untrained)"
MODEL_NAME_CNN = "TrainedCNN v1 (weights/deepfake_cnn.pth)"

# -------------------- individual signal functions --------------------

def _frequency_artifact_ratio(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    y, x = np.ogrid[:h, :w]
    low_mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    low_energy = magnitude[low_mask].sum()
    total_energy = magnitude.sum() + 1e-8
    return float(1.0 - (low_energy / total_energy))


def _noise_variance(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _edge_strength(bgr):
    edges = cv2.Canny(bgr, 100, 200)
    return float(edges.mean())


def _color_consistency(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return float(ycrcb[:, :, 1:].std())


def _normalize(value, lo, hi, invert=False):
    """Map a raw measurement onto a 0-1 'suspicious' scale."""
    span = hi - lo
    if span == 0:
        return 0.0
    score = (value - lo) / span
    score = min(max(score, 0.0), 1.0)
    return (1.0 - score) if invert else score


def score_face_crop(bgr_face):
    face = cv2.resize(bgr_face, (256, 256))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    freq_raw = _frequency_artifact_ratio(gray)
    noise_raw = _noise_variance(gray)
    edge_raw = _edge_strength(face)
    color_raw = _color_consistency(face)

    # Thresholds tuned against typical natural-photo ranges. These are
    # heuristic cutoffs, not calibrated against a labeled dataset.
    signals = {
        "frequency_artifact": round(_normalize(freq_raw, 0.55, 0.80), 3),
        "noise_consistency": round(_normalize(noise_raw, 0, 80, invert=True), 3),
        "edge_sharpness": round(_normalize(edge_raw, 0, 15, invert=True), 3),
        "color_consistency": round(_normalize(color_raw, 0, 8, invert=True), 3),
    }
    weights = {
        "frequency_artifact": 0.35,
        "noise_consistency": 0.30,
        "edge_sharpness": 0.20,
        "color_consistency": 0.15,
    }
    combined = sum(signals[k] * weights[k] for k in signals)
    return combined, signals


def _detect_largest_face(bgr_img):
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None, 0
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad = int(0.2 * w)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(bgr_img.shape[1], x + w + pad)
    y1 = min(bgr_img.shape[0], y + h + pad)
    return bgr_img[y0:y1, x0:x1], len(faces)


# -------------------- public entry points --------------------

def analyze_image_bytes(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image file")

    crop, n_faces = _detect_largest_face(img)
    if crop is None or crop.size == 0:
        crop = img  # no face found: fall back to whole-frame analysis

    combined, signals = score_face_crop(crop)
    return {
        "frame_count": 1,
        "faces_detected": n_faces,
        "per_frame_scores": [round(combined, 3)],
        "signals": signals,
        "combined_score": combined,
        "temporal_variance": None,
    }


def analyze_video_bytes(video_bytes, sample_fps=2, max_frames=40):
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(int(fps / sample_fps), 1)

        per_frame_scores = []
        signal_accum = {"frequency_artifact": 0.0, "noise_consistency": 0.0,
                         "edge_sharpness": 0.0, "color_consistency": 0.0}
        faces_total = 0
        frames_analyzed = 0
        idx = 0

        while frames_analyzed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                crop, n_faces = _detect_largest_face(frame)
                if crop is None or crop.size == 0:
                    crop = frame
                else:
                    faces_total += 1
                combined, signals = score_face_crop(crop)
                per_frame_scores.append(round(combined, 3))
                for k in signal_accum:
                    signal_accum[k] += signals[k]
                frames_analyzed += 1
            idx += 1
        cap.release()
    finally:
        os.unlink(tmp_path)

    if frames_analyzed == 0:
        raise ValueError("Could not extract any frames from this video")

    combined_score = sum(per_frame_scores) / len(per_frame_scores)
    signals_avg = {k: round(v / frames_analyzed, 3) for k, v in signal_accum.items()}
    temporal_variance = float(np.var(per_frame_scores)) if len(per_frame_scores) > 1 else 0.0

    return {
        "frame_count": frames_analyzed,
        "faces_detected": faces_total,
        "per_frame_scores": per_frame_scores,
        "signals": signals_avg,
        "combined_score": combined_score,
        "temporal_variance": round(temporal_variance, 4),
    }
