"""
Loaded by main.py only when trained Keras weights exist at:
  backend/weights/xception_deepfake_image.h5   (from train_image.py)
  backend/weights/video_gru.keras              (from train_video.py)

Mirrors the architectures/preprocessing in train_image.py / train_video.py
exactly, so predictions match what was trained.
"""

import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

import detector

_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
IMAGE_MODEL_PATH = os.path.join(_WEIGHTS_DIR, "xception_deepfake_image.h5")
VIDEO_MODEL_PATH = os.path.join(_WEIGHTS_DIR, "video_gru.keras")

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

_image_model = None
_video_model = None
_feature_extractor = None

def _load_image_model():
    global _image_model

    print("Loading model...")

    if _image_model is None:
        _image_model = keras.models.load_model(
            IMAGE_MODEL_PATH,
            compile=False
        )

    print("Model loaded successfully")

    return _image_model


def _load_video_model():
    global _video_model, _feature_extractor
    if _video_model is None:
        _video_model = keras.models.load_model(
            VIDEO_MODEL_PATH,
            compile=False
        )
        _feature_extractor = keras.applications.InceptionV3(
            weights="imagenet", include_top=False, pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
    return _video_model, _feature_extractor


def score_image(content_bytes):
    arr = np.frombuffer(content_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image file")

    crop, n_faces = detector._detect_largest_face(img)
    if crop is None or crop.size == 0:
        crop = img

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
    crop_rgb = tf.keras.applications.xception.preprocess_input(crop_rgb)

    model = _load_image_model()
    score = float(model.predict(crop_rgb[None, ...], verbose=0)[0][0])

    return {
        "frame_count": 1,
        "faces_detected": n_faces,
        "per_frame_scores": [round(score, 3)],
        "signals": {"cnn_confidence": round(score, 3)},
        "combined_score": score,
        "temporal_variance": None,
    }


def _crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def score_video(content_bytes):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")

        frames = []
        while len(frames) < MAX_SEQ_LENGTH:
            ret, frame = cap.read()
            if not ret:
                break
            frame = _crop_center_square(frame)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            frames.append(frame)
        cap.release()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise ValueError("Could not extract any frames from this video")

    video_model, feature_extractor = _load_video_model()

    length = min(MAX_SEQ_LENGTH, len(frames))
    frame_features = np.zeros((1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    frame_mask = np.zeros((1, MAX_SEQ_LENGTH), dtype="bool")

    per_frame_scores = []
    for j in range(length):
        feat = feature_extractor.predict(np.array(frames)[None, j], verbose=0)
        frame_features[0, j, :] = feat
    frame_mask[0, :length] = 1

    combined_score = float(video_model.predict([frame_features, frame_mask], verbose=0)[0][0])

    # The GRU model scores the whole sequence at once (temporal info is
    # baked into the recurrent layers), so we don't get a natural per-frame
    # score the way the heuristic analyzer does. We report the single
    # sequence-level score for every sampled frame so the dashboard's
    # timeline still renders meaningfully.
    per_frame_scores = [round(combined_score, 3)] * length

    return {
        "frame_count": length,
        "faces_detected": None,
        "per_frame_scores": per_frame_scores,
        "signals": {"cnn_confidence": round(combined_score, 3)},
        "combined_score": combined_score,
        "temporal_variance": None,
    }
