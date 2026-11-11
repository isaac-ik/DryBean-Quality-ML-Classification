"""
Checks behind the numbers in DESIGN_NOTES.md. Run after train_model.py:

    python validate.py

1. Feature formulas reproduce every derived column of the Dry Bean dataset.
2. Camera measurement: 400 bean-like shapes rendered the way the conveyor camera
   would see them (about 4x fewer pixels than the dataset camera). Compares the
   features from the camera pipeline with the exact geometry, and whether the
   model predicts the same variety from both.
3. Scale calibration recovers a known camera scale.
4. A stone-shaped blob is rejected as "unknown" instead of being given a variety.
"""
import json
import math
import os
import tempfile

import cv2
import joblib
import numpy as np

import bean_features as bf
import sorter
import train_model

S = 4.0                                   # dataset px per camera px used in the simulation
BELT_BGR = (160, 95, 35)                  # blue belt


def render(L, l, ang, notch, scale, size):
    """Bean outline (ellipse, optionally kidney-notched) rendered like a camera:
    drawn 4x finer, area-averaged down, thresholded at 50 %. Returns contour and exact polygon."""
    ss, (H, W) = 4, size
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    pts = np.c_[L / 2 / scale * np.cos(t), l / 2 / scale * np.sin(t)]
    if notch:
        r, cy = l / scale * 0.35, l / 2 / scale * 1.05
        pts = pts[np.hypot(pts[:, 0], pts[:, 1] - cy) > r]
    a = math.radians(ang)
    pts = pts @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + [W / 2, H / 2]
    big = np.zeros((H * ss, W * ss), np.uint8)
    cv2.fillPoly(big, [np.round(pts * ss * 256).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=8)
    img = (cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA) >= 128).astype(np.uint8) * 255
    c = max(cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    return c, (pts * scale).astype(np.float32)


def exact_features(poly):
    P = cv2.arcLength(poly.reshape(-1, 1, 2), True)
    A = cv2.contourArea(poly)
    m = cv2.moments(poly)
    a, b, d = m["mu20"] / m["m00"], m["mu11"] / m["m00"], m["mu02"] / m["m00"]
    cm = math.sqrt((a - d) ** 2 + 4 * b * b)
    L = 2 * math.sqrt(2) * math.sqrt(a + d + cm)
    l = 2 * math.sqrt(2) * math.sqrt(a + d - cm)
    bbox = float(np.ptp(poly[:, 0]) * np.ptp(poly[:, 1]))
    return bf.derive(A, P, L, l, cv2.contourArea(cv2.convexHull(poly)), bbox)


def main():
    X, y = train_model.load_arff(train_model.DEFAULT_DATA)
    bundle = joblib.load(train_model.OUT)
    model = bundle["full"]["model"]

    print("1) feature formulas vs dataset")
    worst = 0.0
    for row in X:
        f = bf.derive(row[0], row[1], row[2], row[3], row[6], row[0] / row[8])
        worst = max(worst, max(abs(f[n] - row[i]) / abs(row[i]) for i, n in enumerate(bf.FEATURES)))
    print(f"   {len(X)} rows, worst relative error {worst:.1e}")

    print("2) camera measurement vs exact geometry (400 simulated beans)")
    rng = np.random.default_rng(1)
    agree, errs = 0, {k: [] for k in ("Area", "Perimeter", "MajorAxisLength", "Solidity", "roundness")}
    for _ in range(400):
        r = rng.integers(len(y))
        c, poly = render(X[r, 2], X[r, 3], float(rng.uniform(0, 180)), bool(rng.random() < 0.5), S, (260, 260))
        truth, meas = exact_features(poly), bf.rescale(bf.measure(c), S)
        for k in errs:
            errs[k].append((meas[k] - truth[k]) / truth[k])
        agree += model.predict(bf.vector(meas))[0] == model.predict(bf.vector(truth))[0]
    print("   median error " + ", ".join(f"{k} {np.median(v):+.2%}" for k, v in errs.items()))
    print(f"   same variety predicted: {agree}/400 ({agree / 4:.1f} %)")

    print("3) scale calibration")
    cfg = sorter.load_config()
    with tempfile.TemporaryDirectory() as tmp:
        sorter.SCALE_PATH = os.path.join(tmp, "scale.json")
        cal = sorter.ScaleCalibration(cfg, "DERMASON", 30)
        rows = rng.choice(np.where(y == "DERMASON")[0], 30, replace=False)
        for r in rows:
            c, _ = render(X[r, 2], X[r, 3], float(rng.uniform(0, 180)), False, S, (260, 260))
            cal.add({"dry_bean": bf.measure(c)})
        with open(sorter.SCALE_PATH, encoding="utf-8") as f:
            print(f"   true scale {S}, calibrated {json.load(f)['scale']:.3f}")

        print("4) non-bean object")
        with open(sorter.SCALE_PATH, "w", encoding="utf-8") as f:
            json.dump({"scale": S}, f)
        clf = sorter.Classifier(cfg)
        belt = {"mean": cv2.cvtColor(np.uint8([[BELT_BGR]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float).tolist()}
        frame = np.full((720, 1280, 3), BELT_BGR, np.uint8)
        stone = np.array([[600, 380], [700, 360], [690, 395], [720, 450], [640, 430], [610, 470]], np.int32)
        cv2.fillPoly(frame, [stone], (120, 120, 120))
        print("   stone-shaped blob ->", clf(sorter.find_beans(frame, belt, cfg)[0])[:2])


if __name__ == "__main__":
    main()
