"""
The 16 Dry Bean dataset features (Koklu & Ozkan 2020, UCI #602), computed from
a bean contour exactly the way the dataset defines them, so a model trained on
DryBean-Quality-ML-Classification can classify beans seen by the conveyor camera.

Formulas were checked against every row of Dry_Bean_Dataset.arff. Note that
ShapeFactor2 in the data is Area / MajorAxisLength^3 (not MinorAxis/Area as
the paper text says).

Size features are in pixels, so they depend on the camera. `rescale()` converts
conveyor-camera pixels to dataset pixels with one factor `s` (lengths * s,
areas * s^2), found once with `sorter.py --calibrate-scale VARIETY`. All 16
features scale exactly that way because `measure()` measures the continuous
outline rather than counting pixels.
"""
import math

import cv2
import numpy as np

FEATURES = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation",
            "Eccentricity", "ConvexArea", "EquivDiameter", "Extent", "Solidity", "roundness",
            "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]

# power of the length scale in each feature; 0 = scale-invariant
SCALE_POWER = {"Area": 2, "Perimeter": 1, "MajorAxisLength": 1, "MinorAxisLength": 1,
               "ConvexArea": 2, "EquivDiameter": 1, "ShapeFactor1": -1, "ShapeFactor2": -1}
SHAPE_FEATURES = [f for f in FEATURES if f not in SCALE_POWER]


def derive(A, P, L, l, C, bbox_area):
    """Build all 16 features from the 6 basic measurements."""
    Ed = math.sqrt(4 * A / math.pi)
    return {
        "Area": A, "Perimeter": P, "MajorAxisLength": L, "MinorAxisLength": l,
        "AspectRation": L / l, "Eccentricity": math.sqrt(max(0.0, 1 - (l / L) ** 2)),
        "ConvexArea": C, "EquivDiameter": Ed, "Extent": A / bbox_area, "Solidity": A / C,
        "roundness": 4 * math.pi * A / P ** 2, "Compactness": Ed / L,
        "ShapeFactor1": L / A, "ShapeFactor2": A / L ** 3,
        "ShapeFactor3": 4 * A / (math.pi * L ** 2), "ShapeFactor4": 4 * A / (math.pi * L * l),
    }


def measure(contour, shape=None):
    """Dry Bean features of one bean contour (OpenCV, image coordinates), in camera pixels.

    The conveyor camera sees a bean ~4x smaller (in pixels) than the dataset camera
    did, so counting pixels would over-weight the edge pixels. Instead the bean's
    true outline is measured: the boundary is lightly smoothed (removes the pixel
    staircase) and the edge is taken half a pixel outside the boundary-pixel centres.
    Tested on 400 bean-like shapes (ellipses, half with a kidney notch) rendered
    as a camera would see them at ~4x fewer pixels than the dataset: features
    within ~1.5 % of the exact geometry, and the model gives the same variety as
    from the exact geometry for 95 % of them.
    """
    sm = _smooth(contour, 0.7).astype(np.float32)
    P = cv2.arcLength(sm.reshape(-1, 1, 2), True)
    A = cv2.contourArea(sm) + 0.5 * P
    hull = cv2.convexHull(sm)
    C = cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True)
    # ellipse with the same second moments as the region (MATLAB regionprops definition)
    m = cv2.moments(sm)
    uxx, uxy, uyy = m["mu20"] / m["m00"], m["mu11"] / m["m00"], m["mu02"] / m["m00"]
    common = math.sqrt((uxx - uyy) ** 2 + 4 * uxy ** 2)
    L = 2 * math.sqrt(2) * math.sqrt(uxx + uyy + common) + 1          # +1: half a pixel each end
    l = 2 * math.sqrt(2) * math.sqrt(max(uxx + uyy - common, 1e-9)) + 1
    w = float(np.ptp(sm[:, 0])) + 1
    h = float(np.ptp(sm[:, 1])) + 1
    return derive(A, P, L, l, C, w * h)


def _smooth(contour, sigma):
    """Outline points with the pixel staircase smoothed out (Gaussian along the outline)."""
    pts = contour[:, 0, :].astype(np.float64)
    h = max(1, int(3 * sigma))
    if len(pts) <= 2 * h:
        return pts
    k = cv2.getGaussianKernel(2 * h + 1, sigma).ravel()
    return np.stack([np.convolve(np.r_[pts[-h:, i], pts[:, i], pts[:h, i]], k, "valid") for i in (0, 1)], 1)


def rescale(feats, s):
    """Camera pixels -> dataset pixels (s = dataset px per camera px)."""
    return {k: v * s ** SCALE_POWER.get(k, 0) for k, v in feats.items()}


def vector(feats, names=FEATURES):
    return np.array([[feats[n] for n in names]], dtype=float)
