"""
Train an image deepfake classifier using Xception transfer learning.

Ported from the reference notebook's approach (deep-fake-detection-on-images-
and-videos.ipynb): fine-tune an ImageNet-pretrained Xception backbone on
real/fake face crops. In the original notebook, on the Kaggle
"deepfake-faces" dataset, this reached ~82% test accuracy after two-stage
fine-tuning. Expect similar territory on a similarly-sized, similarly-clean
dataset — not higher "perfect" accuracy. Real-world accuracy depends
heavily on how the deepfakes in your dataset were generated and how close
they are to what you'll actually be scanning.

------------------------------------------------------------------
STEP 1 — Get labeled face-crop images (224x224 works well, matching
  Xception's typical resolution) in this layout:

    dataset/
      train/
        real/   *.jpg
        fake/   *.jpg
      val/
        real/   *.jpg
        fake/   *.jpg

  Sources: Kaggle "deepfake-faces" dataset (pre-cropped faces from the
  DFDC challenge), or crop faces yourself from FaceForensics++/DFDC/
  Celeb-DF frames using detector.py's `_detect_largest_face`.

STEP 2 — Install training deps (heavy, not in the main requirements.txt):

    pip install -r requirements-train.txt

STEP 3 — Run:

    python train_image.py --data_dir ./dataset --out weights/xception_deepfake_image.h5

STEP 4 — main.py will automatically pick up weights/xception_deepfake_image.h5
  and use it instead of the heuristic analyzer / other trained models.
------------------------------------------------------------------

Two-stage fine-tuning, same as the notebook:
  Stage 1: freeze the Xception backbone, train only the new classifier head.
  Stage 2: unfreeze the top blocks of Xception, continue training at a
           lower learning rate.
"""

import argparse
import os

import tensorflow as tf


def build_datasets(data_dir, img_size=224, batch_size=32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(img_size, img_size), batch_size=batch_size,
        label_mode="binary", class_names=["real", "fake"], shuffle=True, seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=(img_size, img_size), batch_size=batch_size,
        label_mode="binary", class_names=["real", "fake"], shuffle=False,
    )

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(mode="horizontal", seed=42),
        tf.keras.layers.RandomRotation(factor=0.05, seed=42),
        tf.keras.layers.RandomContrast(factor=0.2, seed=42),
    ])
    preprocess = tf.keras.applications.xception.preprocess_input

    def prep_train(x, y):
        x = tf.cast(x, tf.float32)
        x = data_augmentation(x, training=True)
        return preprocess(x), y

    def prep_val(x, y):
        return preprocess(tf.cast(x, tf.float32)), y

    train_ds = train_ds.map(prep_train).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.map(prep_val).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds


def build_model():
    tf.random.set_seed(42)
    base_model = tf.keras.applications.xception.Xception(
        weights="imagenet", include_top=False
    )
    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(avg)
    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    return model, base_model


def train(args):
    train_ds, val_ds = build_datasets(args.data_dir, batch_size=args.batch_size)
    model, base_model = build_model()

    # Stage 1: frozen backbone, train the new head only.
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        metrics=["accuracy"],
    )
    print("Stage 1: training classifier head (backbone frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=args.head_epochs)

    # Stage 2: unfreeze top blocks, fine-tune at a lower learning rate.
    for layer in base_model.layers[56:]:
        layer.trainable = True
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=["accuracy"],
    )
    print("Stage 2: fine-tuning top backbone layers")
    model.fit(train_ds, validation_data=val_ds, epochs=args.finetune_epochs)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"Final validation accuracy: {val_acc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)
    print(f"Saved model to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--head_epochs", type=int, default=3)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out", default="weights/xception_deepfake_image.h5")
    args = parser.parse_args()
    train(args)
