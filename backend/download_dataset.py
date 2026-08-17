"""
One-time helper: downloads the 140k Real and Fake Faces dataset via kagglehub
and prints the local path it landed at, so you can pass it straight to
--images_root in train_image_csv.py / evaluate.py.

Usage:
    pip install kagglehub
    python download_dataset.py

First run may open a browser window asking you to log in to Kaggle and
authorize kagglehub — that's expected, it's how kagglehub authenticates
without needing a manually placed kaggle.json API key.
"""

import os
import kagglehub

path = kagglehub.dataset_download("xhlulu/140k-real-and-fake-faces")
print(f"\nDataset downloaded to: {path}\n")

# Quick structure check so you don't have to guess the nesting level.
print("Top-level contents:")
for entry in sorted(os.listdir(path)):
    print(f"  {entry}")

# Try to find the folder that directly contains train/valid/test.
for root, dirs, files in os.walk(path):
    if {"train", "valid", "test"}.issubset(set(dirs)):
        print(f"\n>>> Use this as --images_root:\n{root}\n")
        break
else:
    print("\nCould not auto-locate a folder containing train/valid/test — "
          "inspect the structure above manually.")
