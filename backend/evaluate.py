"""
Evaluate the trained image model on the held-out test.csv — real, honest metrics,
not just accuracy (accuracy alone can hide a model that's bad at catching fakes
specifically, e.g. if it just learned to guess "real" most of the time).

Usage:
    python evaluate.py --images_root "/path/to/real-vs-fake"
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

from train_image_csv import load_split, build_dataset, IMG_SIZE

_HERE = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(_HERE, "weights", "xception_deepfake_image.h5")


def evaluate(args):
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(
            f"No trained model found at {WEIGHTS_PATH}. Run train_image_csv.py first."
        )

    print(f"Loading test.csv and verifying files exist under {args.images_root} ...")
    test_df = load_split("test.csv", args.images_root)
    print(f"test: {len(test_df)} images")

    test_ds = build_dataset(test_df, batch_size=32, training=False)

    print("Loading model...")
    model = tf.keras.models.load_model(WEIGHTS_PATH)

    print("Running predictions on the held-out test set (this is the real number)...")
    probs = model.predict(test_ds, verbose=1).ravel()
    y_true = test_df["binary_label"].values.astype(int)
    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print("TEST SET RESULTS (held-out, never seen during training)")
    print("=" * 50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}  (of predicted fakes, how many were actually fake)")
    print(f"Recall:    {rec:.4f}  (of actual fakes, how many were caught)")
    print(f"F1 score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("\nConfusion matrix:")
    print("                 predicted real   predicted fake")
    print(f"actual real      {cm[0][0]:<16} {cm[0][1]}")
    print(f"actual fake      {cm[1][0]:<16} {cm[1][1]}")
    print("\n" + classification_report(y_true, y_pred, target_names=["real", "fake"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True,
                         help="Folder whose train/valid/test subfolders match the CSVs' path column")
    args = parser.parse_args()
    evaluate(args)
