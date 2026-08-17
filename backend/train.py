"""
Train a real deepfake-detection CNN.

This is NOT runnable inside a sandbox with no internet/dataset access —
it's meant for YOUR machine (ideally with a GPU) once you have a labeled
face dataset. This is the only legitimate way to get materially higher
accuracy than the forensic heuristics in detector.py.

------------------------------------------------------------------
STEP 1 — Get a labeled dataset (pick one; all are free for research use,
but each requires you to individually agree to their license/terms):

  - FaceForensics++  : https://github.com/ondyari/FaceForensics
  - DFDC (Facebook)  : https://ai.meta.com/datasets/dfdc/
  - Celeb-DF v2       : https://github.com/yuezunli/celeb-deepfakeforensics

STEP 2 — Extract faces from every video into two folders of still images:

    dataset/
      train/
        real/   *.jpg
        fake/   *.jpg
      val/
        real/   *.jpg
        fake/   *.jpg

  Use detector.py's `_detect_largest_face` (or MTCNN/RetinaFace for
  higher-quality crops) to crop faces out of each frame before saving.

STEP 3 — Install training deps (NOT in requirements.txt, since they're
  heavy and only needed for training, not for serving):

    pip install torch torchvision

STEP 4 — Run:

    python train.py --data_dir ./dataset --epochs 15 --out weights/deepfake_cnn.pth

STEP 5 — Drop the resulting weights/deepfake_cnn.pth into backend/weights/.
  main.py will detect the file and automatically switch from the
  heuristic analyzer to this trained model (see cnn_inference.py).
------------------------------------------------------------------
"""

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


def build_model():
    # ResNet18 backbone, initialized from scratch (weights=None) because
    # this sandbox has no path to download ImageNet-pretrained weights.
    # On your own machine you should instead use:
    #   models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # which will converge much faster and reach higher accuracy via
    # transfer learning.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)  # [real, fake]
    return model


def get_loaders(data_dir, batch_size, img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)

    # ImageFolder sorts classes alphabetically: expect ['fake', 'real']
    assert train_ds.classes == ["fake", "real"], (
        f"Expected classes ['fake','real'], got {train_ds.classes}. "
        "Rename your folders so 'fake' sorts before 'real'."
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_loader, val_loader = get_loaders(args.data_dir, args.batch_size)
    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / max(total, 1)

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  -> saved new best model ({val_acc:.4f}) to {args.out}")

    print(f"Done. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to dataset/ (with train/ and val/ subfolders)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="weights/deepfake_cnn.pth")
    args = parser.parse_args()
    train(args)
