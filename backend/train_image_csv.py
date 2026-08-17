"""
Train the image deepfake classifier directly from train.csv / valid.csv / test.csv
(dataset_meta/), matching the reference notebook's Xception approach exactly.

This is the "140k Real and Fake Faces" dataset (FFHQ real faces vs GAN-generated
fakes, 100k/20k/20k train/valid/test split, perfectly balanced). The CSVs here are
just the index — id, label, label_str, path — pointing at image files. THE IMAGE
FILES THEMSELVES ARE NOT INCLUDED (it's a 4GB+ download); you need to download and
unzip the dataset yourself, then point --images_root at the folder whose
train/, valid/, test/ subfolders match the `path` column exactly.

Download: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
(unzips to real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/*.jpg —
that whole real-vs-fake folder is what you pass as --images_root)

------------------------------------------------------------------
STEP 1 — Download & unzip the dataset above.

STEP 2 — Install training deps:
    pip install -r requirements-train.txt

STEP 3 — Run (this verifies every file exists before spending any GPU time):
    python train_image_csv.py --images_root "/path/to/real-vs-fake" \
        --out weights/xception_deepfake_image.h5

STEP 4 — Evaluate on the untouched test set:
    python evaluate.py --images_root "/path/to/real-vs-fake"

STEP 5 — Restart main.py. It auto-detects weights/xception_deepfake_image.h5
  and switches the whole API from heuristic mode to this trained model.
------------------------------------------------------------------

Two-stage fine-tuning, identical to the notebook:
  Stage 1: freeze Xception backbone, train only the new classifier head.
  Stage 2: unfreeze layers[56:], continue at a lower learning rate.
"""

import argparse
import os

import pandas as pd
import tensorflow as tf

_HERE = os.path.dirname(__file__)
IMG_SIZE = 224


def load_split(csv_name, images_root):
    """Read a CSV (id,label,label_str,path), resolve full paths, verify files exist."""
    csv_path = os.path.join(_HERE, "dataset_meta", csv_name)
    df = pd.read_csv(csv_path)
    df["full_path"] = df["path"].apply(lambda p: os.path.join(images_root, p))

    missing = ~df["full_path"].apply(os.path.exists)
    n_missing = missing.sum()
    if n_missing > 0:
        sample = df.loc[missing, "full_path"].head(3).tolist()
        raise FileNotFoundError(
            f"{n_missing}/{len(df)} files from {csv_name} not found under "
            f"--images_root '{images_root}'. Example missing paths:\n  "
            + "\n  ".join(sample)
            + "\n\nCheck that --images_root points at the folder whose train/valid/test "
              "subfolders match the CSV's `path` column exactly (e.g. it should contain "
              f"'{df['path'].iloc[0].split('/')[0]}/real/...' and '.../fake/...')."
        )

    # label_str is 'real'/'fake'; label is already 1/0 in these CSVs, but we
    # derive fresh from label_str to be safe against ordering changes.
    df["binary_label"] = (df["label_str"] == "fake").astype("float32")
    return df


def build_dataset(df, batch_size, training, augment_layer=None):
    paths = df["full_path"].values
    labels = df["binary_label"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(df), 10000), seed=42)

    preprocess = tf.keras.applications.xception.preprocess_input

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training and augment_layer is not None:
        def _augment(x, y):
            x = augment_layer(x, training=True)
            return x, y
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    def _prep(x, y):
        return preprocess(tf.cast(x, tf.float32)), y

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model():
    tf.random.set_seed(42)
    base_model = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(avg)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model, base_model


def train(args):
    print(f"Loading CSV index and verifying files exist under {args.images_root} ...")
    train_df = load_split("train.csv", args.images_root)
    valid_df = load_split("valid.csv", args.images_root)
    print(f"train: {len(train_df)} images  |  valid: {len(valid_df)} images")

    if args.sample_size and args.sample_size < len(train_df):
        train_df = (
            train_df.groupby("label_str", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.sample_size // 2), random_state=42))
        )
        print(f"Using a balanced subsample: {len(train_df)} training images "
              f"(pass --sample_size 0 to use the full 100k set)")

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
        tf.keras.layers.RandomRotation(factor=0.05, seed=42),
        tf.keras.layers.RandomContrast(factor=0.2, seed=42),
    ])

    train_ds = build_dataset(train_df, args.batch_size, training=True, augment_layer=data_augmentation)
    valid_ds = build_dataset(valid_df, args.batch_size, training=False)

    model, base_model = build_model()

    # Stage 1: frozen backbone, train the new head only (matches notebook).
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        metrics=["accuracy"],
    )
    print("Stage 1: training classifier head (backbone frozen)")
    model.fit(train_ds, validation_data=valid_ds, epochs=args.head_epochs)

    # Stage 2: unfreeze top blocks, fine-tune at a lower learning rate.
    for layer in base_model.layers[56:]:
        layer.trainable = True
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=["accuracy"],
    )
    print("Stage 2: fine-tuning top backbone layers")
    model.fit(train_ds, validation_data=valid_ds, epochs=args.finetune_epochs)

    val_loss, val_acc = model.evaluate(valid_ds)
    print(f"Final validation accuracy: {val_acc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")
    print("Run evaluate.py next for precision/recall/F1/ROC-AUC on the held-out test set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True,
                         help="Folder whose train/valid/test subfolders match the CSVs' path column")
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sample_size", type=int, default=16000,
                         help="Balanced subsample of the 100k training set for faster iteration. "
                              "Set to 0 to use the full training set.")
    parser.add_argument("--out", default="weights/xception_deepfake_image.h5")
    args = parser.parse_args()
    train(args)
