"""
Train the dry-bean variety model used by sorter.py.

Same data and approach as https://github.com/isaac-ik/DryBean-Quality-ML-Classification
(16 shape features -> 7 varieties; standardised features, gradient boosting was the
best model there), saved as one file the conveyor can load.

    pip install scikit-learn numpy
    python train_model.py                          # uses the repo's DryBeanDataset/ (or downloads it)
    python train_model.py --data path/to/Dry_Bean_Dataset.arff
    python train_model.py --model lgbm             # LightGBM, like PyCaret's best (pip install lightgbm)
    python train_model.py --extra my_beans.csv     # add your own labelled beans (from sorter.py --log)

Two models go into models/drybean_model.joblib:
  full        all 16 features. Needs the camera scale calibrated (sorter.py --calibrate-scale).
  shape_only  the 8 scale-free features. Works straight away, a bit less accurate.
Each also stores an "unknown object" check (Mahalanobis distance to the nearest
variety), so stones, chaff or broken beans are not forced into a variety.
"""
import argparse
import csv
import os
import urllib.request

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bean_features import FEATURES, SHAPE_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_URL = ("https://github.com/isaac-ik/DryBean-Quality-ML-Classification/raw/main/"
            "DryBeanDataset/Dry_Bean_Dataset.arff")
# inside the DryBean-Quality-ML-Classification repo: <repo>/conveyor-sorter/vision -> <repo>/DryBeanDataset
REPO_DATA = os.path.join(HERE, "..", "..", "DryBeanDataset", "Dry_Bean_Dataset.arff")
DEFAULT_DATA = REPO_DATA if os.path.exists(REPO_DATA) else os.path.join(HERE, "data", "Dry_Bean_Dataset.arff")
OUT = os.path.join(HERE, "models", "drybean_model.joblib")


def load_arff(path):
    X, y = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not in_data:
                in_data = line.upper().startswith("@DATA")
                continue
            if line and not line.startswith("%"):
                parts = line.split(",")
                X.append([float(v) for v in parts[:16]])
                y.append(parts[16].strip())
    return np.array(X), np.array(y)


def load_csv(path):
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Class", "").strip():
                X.append([float(row[n]) for n in FEATURES])
                y.append(row["Class"].strip().upper())
    return np.array(X).reshape(-1, 16), np.array(y)


def make_model(kind):
    if kind == "lgbm":
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=123)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=400, random_state=123, n_jobs=-1)
    elif kind == "lr":
        clf = LogisticRegression(C=6.7, max_iter=3000)
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=123)
    return make_pipeline(StandardScaler(), clf)


def novelty_check(Xs, y, classes):
    """Per-variety mean / inverse covariance in standardised space + 99.5 % distance limit."""
    stats = {}
    for c in classes:
        Z = Xs[y == c]
        mu = Z.mean(axis=0)
        cov = np.cov(Z, rowvar=False) + 1e-3 * np.eye(Z.shape[1])
        inv = np.linalg.inv(cov)
        d = np.sqrt(np.einsum("ij,jk,ik->i", Z - mu, inv, Z - mu))
        stats[c] = {"mean": mu, "inv_cov": inv, "limit": float(np.percentile(d, 99.5))}
    return stats


def train(X, y, names, kind):
    idx = [FEATURES.index(n) for n in names]
    Xn = X[:, idx]
    Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.2, random_state=42, stratify=y)
    model = make_model(kind).fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = accuracy_score(yte, pred)
    print(f"\n=== {len(names)} features, {kind}: test accuracy {acc:.4f}")
    print(classification_report(yte, pred, digits=3))
    model = make_model(kind).fit(Xn, y)                 # final model on all data
    scaler = model.steps[0][1]
    return {"model": model, "features": names, "test_accuracy": float(acc),
            "novelty": novelty_check(scaler.transform(Xn), y, model.classes_)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--extra", help="CSV of your own labelled beans (columns = 16 features + Class)")
    ap.add_argument("--model", default="hgb", choices=["hgb", "lgbm", "rf", "lr"])
    args = ap.parse_args()

    if not os.path.exists(args.data):
        os.makedirs(os.path.dirname(args.data), exist_ok=True)
        print("downloading dataset from", REPO_URL)
        urllib.request.urlretrieve(REPO_URL, args.data)
    X, y = load_arff(args.data) if args.data.lower().endswith(".arff") else load_csv(args.data)
    print(f"dataset: {len(y)} beans, varieties: {', '.join(sorted(set(map(str, y))))}")
    if args.extra:
        Xe, ye = load_csv(args.extra)
        print(f"+ {len(ye)} of your own beans from {args.extra}")
        X, y = np.vstack([X, Xe]), np.concatenate([y, ye])

    full = train(X, y, FEATURES, args.model)
    shape = train(X, y, SHAPE_FEATURES, args.model)
    major = X[:, FEATURES.index("MajorAxisLength")]
    bundle = {
        "full": full, "shape_only": shape, "classes": list(full["model"].classes_),
        # used by sorter.py --calibrate-scale to match the camera to the dataset's pixels
        "median_major_axis_px": {c: float(np.median(major[y == c])) for c in full["model"].classes_},
        "source": REPO_URL,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    joblib.dump(bundle, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
