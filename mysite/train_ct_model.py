"""Train the Lung CT classifier (VGG16 transfer learning, 4 classes).

Usage (from the mysite/ directory):
    python train_ct_model.py [data_dir] [output.h5]

data_dir must contain train/ and valid/ with one sub-folder per class. The test
split is never touched here; model selection uses the validation split only.
Preprocessing matches the web app: 128x128 RGB scaled to [0, 1].

Two phases: (1) train a new classifier head on the frozen ImageNet backbone,
(2) fine-tune the last VGG block at a low learning rate. Augmentation and class
weights compensate for the very small, imbalanced training set.
"""
import os
import sys

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (128, 128)
BATCH = 16
SEED = 42


def make_generators(data_dir):
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        shear_range=0.05,
        brightness_range=(0.85, 1.15),
        horizontal_flip=True,
        fill_mode="nearest",
    ).flow_from_directory(
        os.path.join(data_dir, "train"), target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", shuffle=True, seed=SEED,
    )
    valid_gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        os.path.join(data_dir, "valid"), target_size=IMG_SIZE, batch_size=BATCH,
        class_mode="categorical", shuffle=False,
    )
    return train_gen, valid_gen


def build_model(num_classes):
    base = VGG16(weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(base.input, out), base


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "trained_model_v2.h5"
    train_gen, valid_gen = make_generators(data_dir)
    print("Class order:", train_gen.class_indices)

    weights = compute_class_weight("balanced", classes=np.unique(train_gen.classes), y=train_gen.classes)
    class_weight = dict(enumerate(weights))

    model, base = build_model(train_gen.num_classes)
    ckpt = callbacks.ModelCheckpoint(out_path, monitor="val_accuracy", mode="max", save_best_only=True, verbose=1)

    model.compile(optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=valid_gen, epochs=15, class_weight=class_weight, callbacks=[ckpt], verbose=2)

    for layer in base.layers:
        layer.trainable = layer.name.startswith("block5")
    model.compile(optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    stop = callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=8, restore_best_weights=True)
    model.fit(train_gen, validation_data=valid_gen, epochs=40, class_weight=class_weight, callbacks=[ckpt, stop], verbose=2)
    print("Best model saved to", out_path)


if __name__ == "__main__":
    main()
