"""
Train a video deepfake classifier: InceptionV3 frame features -> GRU sequence
model. Ported from the reference notebook's CNN-RNN architecture, which
reached ~80% accuracy on a DFDC sample split. Same accuracy caveat as
train_image.py applies — this gets you in the same territory as the
notebook, not "perfect" accuracy.

------------------------------------------------------------------
STEP 1 — Labeled videos in this layout, with a metadata.json mapping
  filename -> {"label": "REAL"|"FAKE"} (same format as the DFDC challenge's
  train_sample_videos):

    video_dataset/
      videos/
        abc123.mp4
        def456.mp4
        ...
      metadata.json

STEP 2 — Install training deps:

    pip install -r requirements-train.txt

STEP 3 — Run:

    python train_video.py --data_dir ./video_dataset --out weights/video_gru.keras

STEP 4 — main.py will automatically pick up weights/video_gru.keras for
  video analysis once present.
------------------------------------------------------------------
"""

import argparse
import json
import os

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 224
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]


def load_video(path, max_frames=MAX_SEQ_LENGTH, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet", include_top=False, pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def prepare_all_videos(video_dir, filenames, labels, feature_extractor):
    num_samples = len(filenames)
    frame_masks = np.zeros((num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros((num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for idx, fname in enumerate(filenames):
        frames = load_video(os.path.join(video_dir, fname))
        length = min(MAX_SEQ_LENGTH, len(frames))
        for j in range(length):
            frame_features[idx, j, :] = feature_extractor.predict(frames[None, j], verbose=0)
        frame_masks[idx, :length] = 1
        if idx % 20 == 0:
            print(f"  processed {idx}/{num_samples} videos")

    return (frame_features, frame_masks), np.array(labels, dtype="int32")


def build_sequence_model():
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model([frame_features_input, mask_input], output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train(args):
    meta_path = os.path.join(args.data_dir, "metadata.json")
    video_dir = os.path.join(args.data_dir, "videos")
    with open(meta_path) as f:
        metadata = json.load(f)

    filenames = list(metadata.keys())
    labels = [1 if metadata[f]["label"].upper() == "FAKE" else 0 for f in filenames]

    from sklearn.model_selection import train_test_split
    train_files, val_files, train_labels, val_labels = train_test_split(
        filenames, labels, test_size=0.15, random_state=42, stratify=labels
    )

    print("Building InceptionV3 feature extractor...")
    feature_extractor = build_feature_extractor()

    print(f"Extracting features for {len(train_files)} training videos...")
    train_data, train_labels = prepare_all_videos(video_dir, train_files, train_labels, feature_extractor)
    print(f"Extracting features for {len(val_files)} validation videos...")
    val_data, val_labels = prepare_all_videos(video_dir, val_files, val_labels, feature_extractor)

    model = build_sequence_model()
    model.summary()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        args.out, save_best_only=True, monitor="val_accuracy", mode="max"
    )
    model.fit(
        [train_data[0], train_data[1]], train_labels,
        validation_data=([val_data[0], val_data[1]], val_labels),
        callbacks=[checkpoint],
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    val_loss, val_acc = model.evaluate([val_data[0], val_data[1]], val_labels)
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"Best model saved to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder with videos/ and metadata.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out", default="weights/video_gru.keras")
    args = parser.parse_args()
    train(args)
