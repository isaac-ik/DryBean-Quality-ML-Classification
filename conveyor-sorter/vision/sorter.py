"""
Bean sorter - PC side.

Watches the belt through the camera on the mast, finds each bean as it crosses
the trigger line (directly under the camera), classifies it and tells the
Arduino which gate should divert it. The Arduino does the timing itself by
counting motor steps, so the PC only has to say "this bean, gate N, seen X ms ago".

Classification uses the dry-bean variety model from
https://github.com/isaac-ik/DryBean-Quality-ML-Classification (train it once with
train_model.py), plus simple colour rules for defects (black / spotted beans).

    pip install opencv-python numpy pyserial scikit-learn

    python train_model.py                        # once: builds models/drybean_model.joblib
    python sorter.py --calibrate                 # empty, running belt: learns the belt colour
    python sorter.py --calibrate-scale DERMASON  # run ~30 beans of ONE known variety: matches
                                                 # camera pixels to the dataset's pixels
    python sorter.py --port COM5                 # run (Windows: COMx, Linux: /dev/ttyUSB0)
    python sorter.py --dry-run                   # no Arduino, just show detections
    python sorter.py --dry-run --log beans.csv --label SIRA
                                                 # record features of known beans to retrain
                                                 # with your own camera (train_model.py --extra)

Keys in the window: q quit, space pause belt, g resume belt, 1/2/3 test a gate.
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np

import bean_features as bf

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
BELT_PATH = os.path.join(HERE, "belt_calibration.json")
SCALE_PATH = os.path.join(HERE, "scale_calibration.json")


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_json(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ------------------------------------------------------------- detection
def belt_mask(lab, belt, cfg):
    """Pixels that do not look like the belt (print the belt in blue or green PLA)."""
    d = lab.astype(np.float32) - np.array(belt["mean"], np.float32)
    dist = np.sqrt((d[..., 0] * 0.6) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
    mask = (dist > cfg["detect"]["color_distance"]).astype(np.uint8) * 255
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def find_beans(frame, belt, cfg):
    x0, y0, x1, y1 = cfg["roi"]
    roi = frame[y0:y1, x0:x1]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    mask = belt_mask(lab, belt, cfg)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    beans = []
    amin, amax = cfg["detect"]["min_area_px"], cfg["detect"]["max_area_px"]
    h, w = mask.shape
    for c in contours:
        area = cv2.contourArea(c)
        if not amin <= area <= amax:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1:
            continue                                   # cut off by the ROI edge: shape is wrong
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        bean_mask = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(bean_mask, [c], -1, 255, -1)
        beans.append({
            "cx": cx + x0, "cy": cy + y0, "area": area, "contour": c + [x0, y0],
            "box": (bx + x0, by + y0, bw, bh),
            "features": colour_features(roi, lab, bean_mask, c, area),
            "dry_bean": bf.measure(c, mask.shape),
        })
    return beans


def colour_features(bgr, lab, mask, contour, area):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L, A, B = [float(v) for v in cv2.mean(lab, mask)[:3]]
    H, S, V = [float(v) for v in cv2.mean(hsv, mask)[:3]]
    (_, _), (w, h), _ = cv2.minAreaRect(contour)
    aspect = max(w, h) / max(min(w, h), 1e-6)
    # spots / mould: share of pixels much darker than the bean's own mean
    l_px = lab[..., 0][mask > 0].astype(np.float32)
    dark_frac = float(np.mean(l_px < L * 0.65)) if l_px.size else 0.0
    # OpenCV 8-bit Lab: L is 0..255 -> report 0..100; a/b are offset by 128
    return {"L": L * 100 / 255, "a": A - 128, "b": B - 128, "H": H * 2, "S": S / 2.55, "V": V / 2.55,
            "area": area, "aspect": aspect, "dark_frac": dark_frac}


# ----------------------------------------------------------- classification
class Classifier:
    """Colour defect rules first, then the dry-bean variety model, then map to a gate."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cc = cfg["classifier"]
        self.model = None
        self.scale = 1.0
        if self.cc["mode"] == "model":
            import joblib
            path = os.path.join(HERE, self.cc["model_path"])
            if not os.path.exists(path):
                raise SystemExit(f"{path} not found - run: python train_model.py")
            bundle = joblib.load(path)
            cal = load_json(SCALE_PATH)
            if cal:
                self.part, self.scale = bundle["full"], cal["scale"]
                print(f"model: all 16 features, camera scale x{self.scale:.3f} "
                      f"(test accuracy {self.part['test_accuracy']:.1%})")
            else:
                self.part = bundle["shape_only"]
                print(f"model: scale not calibrated yet -> shape-only features "
                      f"(test accuracy {self.part['test_accuracy']:.1%}). "
                      f"Run --calibrate-scale VARIETY for the full model.")
            self.model = self.part["model"]

    def variety_gate(self, variety):
        override = self.cc.get("variety_gates", {})
        if variety in override:
            return override[variety]
        return 0 if variety in self.cc["target_varieties"] else self.cc["other_variety_gate"]

    def dataset_features(self, bean):
        return bf.rescale(bean["dry_bean"], self.scale)

    def __call__(self, bean):
        """Returns (label, gate, confidence)."""
        f = bean["features"]
        for rule in self.cfg["rules"]:
            if all(lo <= f[k] <= hi for k, (lo, hi) in rule["when"].items()):
                return rule["name"], rule["gate"], 1.0
        if self.model is None:
            return self.cfg["default_class"]["name"], self.cfg["default_class"]["gate"], 1.0

        x = bf.vector(self.dataset_features(bean), self.part["features"])
        z = self.model.steps[0][1].transform(x)[0]
        # does it look like any bean variety at all?
        ratio = min(np.sqrt((z - s["mean"]) @ s["inv_cov"] @ (z - s["mean"])) / s["limit"]
                    for s in self.part["novelty"].values())
        if ratio > self.cc["novelty_factor"]:
            return "unknown", self.cc["uncertain_gate"], 0.0
        p = self.model.predict_proba(x)[0]
        i = int(np.argmax(p))
        variety = str(self.model.classes_[i])
        if p[i] < self.cc["min_confidence"]:
            return f"unsure:{variety}", self.cc["uncertain_gate"], float(p[i])
        return variety, self.variety_gate(variety), float(p[i])


# --------------------------------------------------------------- tracking
class Tracker:
    """Very small centroid tracker. Beans only move forward along the belt."""

    def __init__(self, direction, max_jump):
        self.dir = direction
        self.max_jump = max_jump
        self.tracks = {}
        self.next_id = 0

    def update(self, beans, now):
        used = set()
        for tid, t in list(self.tracks.items()):
            best, best_d = None, self.max_jump
            for i, b in enumerate(beans):
                if i in used:
                    continue
                fwd = (b["cx"] - t["cx"]) * self.dir
                d = abs(b["cy"] - t["cy"]) + max(0.0, -fwd) * 4 + abs(fwd) * 0.5
                if fwd > -8 and d < best_d:
                    best, best_d = i, d
            if best is None:
                if now - t["seen"] > 0.5:
                    del self.tracks[tid]
                continue
            used.add(best)
            b = beans[best]
            t.update(prev_cx=t["cx"], cx=b["cx"], cy=b["cy"], seen=now, bean=b)
        for i, b in enumerate(beans):
            if i not in used:
                self.tracks[self.next_id] = {"cx": b["cx"], "prev_cx": b["cx"], "cy": b["cy"],
                                             "seen": now, "bean": b, "fired": False}
                self.next_id += 1
        return self.tracks


# --------------------------------------------------------------- serial
class Board:
    def __init__(self, port, dry):
        self.ser = None
        if not dry:
            import serial
            self.ser = serial.Serial(port, 115200, timeout=0)
            time.sleep(2.0)  # Uno resets when the port opens

    def send(self, line):
        print(">>", line)
        if self.ser:
            self.ser.write((line + "\n").encode())

    def poll(self):
        if self.ser and self.ser.in_waiting:
            for ln in self.ser.read(self.ser.in_waiting).decode(errors="ignore").splitlines():
                if ln.strip():
                    print("<<", ln.strip())


# ----------------------------------------------------------------- main
def open_camera(cfg):
    cap = cv2.VideoCapture(cfg["camera"]["index"], cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    if cfg["camera"].get("lock_exposure"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, cfg["camera"]["exposure"])
    if not cap.isOpened():
        raise SystemExit("camera not found - check config.json camera.index")
    return cap


def calibrate(cfg):
    cap = open_camera(cfg)
    x0, y0, x1, y1 = cfg["roi"]
    samples = []
    print("Calibrating: keep the belt EMPTY and running...")
    t_end = time.time() + 4
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok:
            continue
        lab = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2LAB).reshape(-1, 3)
        samples.append(np.median(lab, axis=0))
        cv2.imshow("calibrate", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    belt = {"mean": np.median(np.array(samples), axis=0).tolist()}
    with open(BELT_PATH, "w", encoding="utf-8") as f:
        json.dump(belt, f, indent=2)
    print("belt colour (OpenCV Lab):", belt["mean"], "->", BELT_PATH)


class ScaleCalibration:
    """Measures beans of one known variety; scale = dataset median length / camera median length."""

    def __init__(self, cfg, variety, n):
        import joblib
        bundle = joblib.load(os.path.join(HERE, cfg["classifier"]["model_path"]))
        variety = variety.upper()
        if variety not in bundle["median_major_axis_px"]:
            raise SystemExit(f"variety must be one of {sorted(bundle['median_major_axis_px'])}")
        self.variety, self.n = variety, n
        self.target = bundle["median_major_axis_px"][variety]
        self.lengths = []

    def add(self, bean):
        self.lengths.append(bean["dry_bean"]["MajorAxisLength"])
        print(f"   {self.variety} #{len(self.lengths)}: length {self.lengths[-1]:.1f} camera px")
        if len(self.lengths) < self.n:
            return False
        s = self.target / float(np.median(self.lengths))
        with open(SCALE_PATH, "w", encoding="utf-8") as f:
            json.dump({"scale": s, "variety": self.variety, "beans": len(self.lengths),
                       "median_camera_px": float(np.median(self.lengths))}, f, indent=2)
        print(f"scale = {s:.4f} dataset px per camera px -> {SCALE_PATH}")
        return True


def run(args, cfg):
    belt = load_json(BELT_PATH)
    if belt is None:
        raise SystemExit("run with --calibrate first (empty belt)")
    scale_cal = ScaleCalibration(cfg, args.calibrate_scale, args.beans) if args.calibrate_scale else None
    classify = None if scale_cal else Classifier(cfg)
    board = Board(args.port, args.dry_run)
    board.send(f"R {cfg['belt_rpm']}")
    board.send("G")
    cap = open_camera(cfg)
    direction = 1 if cfg["belt_moves_right_in_image"] else -1
    line_x = cfg["trigger_line_x"]
    tracker = Tracker(direction, max_jump=cfg["detect"]["max_jump_px"])
    if args.collect:
        os.makedirs(args.collect, exist_ok=True)
    log = None
    if args.log:
        new = not os.path.exists(args.log)
        log_file = open(args.log, "a", newline="", encoding="utf-8")
        log = csv.writer(log_file)
        if new:
            log.writerow(bf.FEATURES + ["Class", "Predicted", "Confidence", "Scale"])
    counts = {}
    done = False
    while not done:
        ok, frame = cap.read()
        t_frame = time.monotonic()
        if not ok:
            continue
        beans = find_beans(frame, belt, cfg)
        tracks = tracker.update(beans, t_frame)
        for t in tracks.values():
            crossed = (t["prev_cx"] - line_x) * direction < 0 <= (t["cx"] - line_x) * direction
            if not crossed or t["fired"]:
                continue
            t["fired"] = True
            bean = t["bean"]
            if scale_cal:
                t["label"] = scale_cal.variety
                done = scale_cal.add(bean)
                continue
            name, gate, conf = classify(bean)
            t["label"] = f"{name} {conf:.0%}" if classify.model else name
            counts[name] = counts.get(name, 0) + 1
            age_ms = int((time.monotonic() - t_frame) * 1000) + cfg["camera"]["latency_ms"]
            board.send(f"D {gate} {age_ms}")
            print(f"   {name:16s} {conf:5.0%}  gate {gate}")
            if log:
                feats = classify.dataset_features(bean)
                log.writerow([f"{feats[n]:.6g}" for n in bf.FEATURES] +
                             [args.label.upper() if args.label else "", name, f"{conf:.3f}", classify.scale])
            if args.collect:
                bx, by, bw, bh = bean["box"]
                crop = frame[max(by - 6, 0):by + bh + 6, max(bx - 6, 0):bx + bw + 6]
                cv2.imwrite(os.path.join(args.collect, f"{int(time.time() * 1000)}_{name.replace(':', '-')}.png"), crop)
        board.poll()

        # overlay
        x0, y0, x1, y1 = cfg["roi"]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 200, 0), 1)
        cv2.line(frame, (line_x, y0), (line_x, y1), (0, 0, 255), 2)
        for t in tracks.values():
            b = t["bean"]
            cv2.drawContours(frame, [b["contour"]], -1, (0, 255, 0) if t["fired"] else (0, 200, 255), 2)
            if t.get("label"):
                cv2.putText(frame, t["label"], (int(b["cx"]) - 30, int(b["box"][1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        y = 22
        for name, n in sorted(counts.items()):
            cv2.putText(frame, f"{name}: {n}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            y += 22
        cv2.imshow("bean sorter", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            board.send("S")
        if key == ord("g"):
            board.send("G")
        if key in (ord("1"), ord("2"), ord("3")):
            board.send(f"T {chr(key)}")
    board.send("S")
    if log:
        log_file.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--dry-run", action="store_true", help="no Arduino attached")
    ap.add_argument("--calibrate", action="store_true", help="learn the empty belt colour")
    ap.add_argument("--calibrate-scale", metavar="VARIETY",
                    help="feed beans of this one variety to match camera pixels to the dataset")
    ap.add_argument("--beans", type=int, default=30, help="beans to measure for --calibrate-scale")
    ap.add_argument("--log", metavar="CSV", help="append each bean's 16 features to a CSV")
    ap.add_argument("--label", metavar="VARIETY", help="true variety of the beans being logged")
    ap.add_argument("--collect", metavar="DIR", help="save each bean crop image")
    a = ap.parse_args()
    c = load_config()
    calibrate(c) if a.calibrate else run(a, c)
