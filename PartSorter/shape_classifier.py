from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

TEMPLATE_CANVAS_SIZE = 96
DEFECT_DEPTH_THRESHOLD = 4.0
DEFECT_PENALTY = 0.09
SIDE_COUNT_PENALTY = 0.035
SHAPE_SCORE_GAP = 0.01
SHAPE_SCORE_GAP_LT = 0.03
COLOR_BLACK_V = 35
COLOR_LOW_SAT = 45


def clean_shape_name(shape_with_score: str) -> str:
    return shape_with_score.split("(")[0].strip() if shape_with_score else "Unknown"


def extract_score(shape_with_score: str) -> Optional[float]:
    if not shape_with_score:
        return None
    i = shape_with_score.find("(")
    j = shape_with_score.find(")", i + 1)
    if i == -1 or j == -1:
        return None
    try:
        return float(shape_with_score[i + 1 : j])
    except Exception:
        return None


def direction(shape_name: str, color_name: str) -> str:
    if shape_name == "Skew" and color_name == "Orange":
        return "left"
    if shape_name == "L" and color_name == "Green":
        return "left"
    if shape_name == "Skew" and color_name == "Green":
        return "right"
    if shape_name == "L" and color_name == "Orange":
        return "right"
    return "outward"


def classify_color(frame: np.ndarray, contour: np.ndarray) -> str:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    hsv = cv2.cvtColor(cv2.bitwise_and(frame, frame, mask=mask), cv2.COLOR_BGR2HSV)

    v = hsv[:, :, 2][mask > 0]
    if v.size == 0:
        return "Unknown"
    if np.mean(v) <= COLOR_BLACK_V:
        return "Black"

    s = hsv[:, :, 1][mask > 0]
    h = hsv[:, :, 0][mask > 0]
    valid = s >= COLOR_LOW_SAT
    if np.count_nonzero(valid) == 0:
        return "White" if np.mean(v) > 190 else "Gray"

    hue = float(np.median(h[valid]))
    sat = float(np.mean(s[valid]))
    val = float(np.mean(v[valid]))
    if val < 50:
        return "Black"
    if sat < 45 and val > 190:
        return "White"
    if sat < 55:
        return "Gray"
    if hue < 6 or hue >= 170:
        return "Red"
    if hue < 15:
        return "Orange"
    if hue < 78:
        return "Green"
    return "Unknown"


def _make_template_contour(blocks: List[Tuple[int, int]]) -> np.ndarray:
    canvas = np.zeros((200, 200), dtype=np.uint8)
    cell, offset = 24, 20
    for bx, by in blocks:
        x0, y0 = offset + bx * cell, offset + by * cell
        cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours[0]


def _rotations(contour: np.ndarray) -> List[np.ndarray]:
    x, y, w, h = cv2.boundingRect(contour)
    center = (x + w / 2.0, y + h / 2.0)
    out = []
    for angle in (0, 90, 180, 270):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        out.append(cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), m).astype(np.float32))
    return out


def _contour_to_canvas(contour: np.ndarray) -> Optional[np.ndarray]:
    if contour is None or len(contour) < 3:
        return None
    points = contour.astype(np.float32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(points.astype(np.int32))
    if w <= 0 or h <= 0:
        return None

    usable = max(1, TEMPLATE_CANVAS_SIZE - 8)
    scale = usable / float(max(w, h))
    tx = 4 - x * scale + max(0.0, (usable - w * scale) / 2.0)
    ty = 4 - y * scale + max(0.0, (usable - h * scale) / 2.0)
    mat = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    norm = cv2.transform(points, mat).reshape(-1, 1, 2)

    canvas = np.zeros((TEMPLATE_CANVAS_SIZE, TEMPLATE_CANVAS_SIZE), dtype=np.uint8)
    cv2.drawContours(canvas, [norm.astype(np.int32)], -1, 255, -1)
    cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), dst=canvas)
    return canvas


def _convex_profile(contour: np.ndarray) -> Tuple[int, float, float]:
    if contour is None or len(contour) < 4:
        return (0, 0.0, 0.0)
    c = contour.astype(np.int32)
    try:
        hull = cv2.convexHull(c, returnPoints=False)
        defects = cv2.convexityDefects(c, hull)
    except Exception:
        return (0, 0.0, 0.0)
    if hull is None or len(hull) < 3 or defects is None or len(defects) == 0:
        return (0, 0.0, 0.0)
    depths = [float(d[0][3]) / 256.0 for d in defects if float(d[0][3]) / 256.0 >= DEFECT_DEPTH_THRESHOLD * 0.25]
    if not depths:
        return (0, 0.0, 0.0)
    arr = np.array(depths, dtype=np.float32)
    return (int(len(arr)), float(np.mean(arr)), float(np.max(arr)))


class ShapeClassifier:
    def __init__(self) -> None:
        t = _make_template_contour([(1, 0), (0, 1), (1, 1), (2, 1)])
        s = _make_template_contour([(0, 0), (1, 0), (1, 1), (2, 1)])
        l = _make_template_contour([(0, 0), (0, 1), (1, 1), (2, 1)])
        self.templates: Dict[str, List[np.ndarray]] = {
            "T": [t] + _rotations(t),
            "Skew": [s] + _rotations(s),
            "L": [l] + _rotations(l),
        }

        self.bitmaps: Dict[str, List[np.ndarray]] = {}
        self.defect_counts: Dict[str, int] = {}
        self.defect_profiles: Dict[str, List[Tuple[int, float, float]]] = {}
        self.side_counts: Dict[str, int] = {}

        for label, vars_ in self.templates.items():
            b, c, p, sd = [], [], [], []
            for contour in vars_:
                b.append(_contour_to_canvas(contour))
                prof = _convex_profile(contour)
                c.append(prof[0])
                p.append(prof)
                perim = cv2.arcLength(contour, True)
                eps = 0.030 * perim if perim > 0 else 0.0
                poly = cv2.approxPolyDP(contour, eps, True) if eps > 0 else contour
                sd.append(len(poly))
            self.bitmaps[label] = b
            self.defect_counts[label] = int(round(np.median(c))) if c else 0
            self.defect_profiles[label] = p
            self.side_counts[label] = int(round(np.median(sd))) if sd else 0

    def classify_shape(self, contour: np.ndarray, score_threshold: float, debug: List[str]) -> str:
        eps = 0.030 * cv2.arcLength(contour, True)
        norm = cv2.approxPolyDP(contour, eps, True)
        side_count = len(norm)
        bitmap = _contour_to_canvas(norm)
        if bitmap is None:
            bitmap = np.zeros((TEMPLATE_CANVAS_SIZE, TEMPLATE_CANVAS_SIZE), dtype=np.uint8)

        candidate_profile = _convex_profile(norm)
        defect_count = candidate_profile[0]

        best_label, second_label = "Unknown", "Unknown"
        best_score, second_score = 1e9, 1e9

        for label, variants in self.templates.items():
            label_best = 1e9
            for i, template in enumerate(variants):
                s = cv2.matchShapes(norm, template, cv2.CONTOURS_MATCH_I1, 0.0)
                tb = self.bitmaps[label][i]
                if tb is not None:
                    diff = cv2.countNonZero(cv2.absdiff(bitmap, tb))
                    s = min(s, diff / float(TEMPLATE_CANVAS_SIZE * TEMPLATE_CANVAS_SIZE))
                label_best = min(label_best, s)

            geom = abs(defect_count - self.defect_counts[label]) * DEFECT_PENALTY
            side = abs(side_count - self.side_counts[label]) * SIDE_COUNT_PENALTY
            combined = label_best + geom + side
            debug.append(f"classify:{label} best={label_best:.3f} geom={geom:.3f} sides={side:.3f} combined={combined:.3f}")

            if combined < second_score:
                if combined < best_score:
                    second_score, second_label = best_score, best_label
                    best_score, best_label = combined, label
                else:
                    second_score, second_label = combined, label

        if best_score > score_threshold:
            return "Unknown"

        def closest_profile(name: str):
            profiles = self.defect_profiles.get(name, [])
            if not profiles:
                return (0, 0.0, 0.0)
            return min(profiles, key=lambda p: abs(defect_count - p[0]) + abs(candidate_profile[0] - p[0]))

        def delta(p):
            return abs(candidate_profile[0] - p[0]) + abs(candidate_profile[1] - p[1]) + abs(candidate_profile[2] - p[2])

        if second_label in ("L", "Skew") and best_label in ("L", "Skew") and (second_score - best_score) < SHAPE_SCORE_GAP:
            if delta(closest_profile(second_label)) < delta(closest_profile(best_label)):
                best_label, best_score = second_label, second_score
        if second_label in ("L", "T") and best_label in ("L", "T") and (second_score - best_score) < SHAPE_SCORE_GAP_LT:
            if delta(closest_profile(second_label)) < delta(closest_profile(best_label)):
                best_label, best_score = second_label, second_score

        return f"{best_label} ({best_score:.3f})"
