"""Reproducible image-classification baseline for ExpW-style annotations.

This CPU baseline is separate from the historical CNN notebook.
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(images, labels):
    root = Path(images).resolve()
    features, targets, groups = [], [], []
    for number, line in enumerate(Path(labels).read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"Invalid annotation on line {number}: expected 8 fields")
        name, _, top, left, right, bottom, confidence, label = fields
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Image path escapes data directory on line {number}")
        top, left, right, bottom, label = map(int, (top, left, right, bottom, label))
        if not 0 <= label <= 6:
            raise ValueError(f"Class outside 0..6 on line {number}")
        with Image.open(path) as image:
            if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
                raise ValueError(f"Invalid face bounds on line {number}")
            crop = image.convert("L").crop((left, top, right, bottom)).resize((32, 32))
            features.append(np.asarray(crop, dtype=np.float32).ravel() / 255.0)
        targets.append(label)
        groups.append(name)
    if len(set(groups)) < 5:
        raise ValueError("At least five distinct source images are required")
    return np.asarray(features), np.asarray(targets), np.asarray(groups)


def train(images, labels, output):
    X, y, groups = load_data(images, labels)
    train_ids, test_ids = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
    if len(set(y[train_ids])) < 2:
        raise ValueError("Training partition needs at least two classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    model.fit(X[train_ids], y[train_ids])
    prediction = model.predict(X[test_ids])
    metrics = {"train_faces": len(train_ids), "test_faces": len(test_ids),
               "balanced_accuracy": balanced_accuracy_score(y[test_ids], prediction),
               "report": classification_report(y[test_ids], prediction, output_dict=True, zero_division=0)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    train(args.images, args.labels, args.output)
