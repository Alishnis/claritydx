"""Evaluate the Lung CT classifier (trained_model.h5) on a held-out test set.

Usage (from the mysite/ directory):
    python evaluate_ct_model.py [path/to/test_dir]

The test directory must contain one sub-folder per class. Folders are sorted
alphabetically, which matches the label order the model was trained with
(adenocarcinoma, large.cell, normal, squamous.cell). Preprocessing mirrors the
web app: resize to 128x128 and scale pixel values to [0, 1].
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

IMG_SIZE = (128, 128)
MODEL_PATH = os.environ.get("CT_MODEL_PATH", "trained_model.h5")
CLASS_NAMES = ["adenocarcinoma", "large.cell.carcinoma", "normal", "squamous.cell.carcinoma"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def collect_samples(test_dir):
    folders = sorted(d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)))
    if len(folders) != len(CLASS_NAMES):
        raise SystemExit(f"Expected {len(CLASS_NAMES)} class folders in {test_dir}, found {folders}")
    samples = []
    for label, folder in enumerate(folders):
        folder_path = os.path.join(test_dir, folder)
        for name in sorted(os.listdir(folder_path)):
            if name.lower().endswith(IMAGE_EXTS):
                samples.append((os.path.join(folder_path, name), label))
    return samples


def main():
    test_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "test")
    samples = collect_samples(test_dir)
    model = load_model(MODEL_PATH)

    y_true, y_pred = [], []
    for path, label in samples:
        img = img_to_array(load_img(path, target_size=IMG_SIZE)) / 255.0
        probs = model.predict(np.expand_dims(img, axis=0), verbose=0)
        y_true.append(label)
        y_pred.append(int(np.argmax(probs, axis=1)[0]))

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred).tolist()
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(json.dumps(matrix))
    print("RESULT_JSON " + json.dumps({"n": len(y_true), "report": report, "confusion_matrix": matrix}))


if __name__ == "__main__":
    main()
