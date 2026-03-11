from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    min_shape_area: int = 120
    max_shape_area_ratio: float = 0.30
    min_shape_area_far: int = 40
    min_shape_area_ratio_near: float = 0.0002
    min_shape_area_ratio_far: float = 0.00008
    calibration_frame_count: int = 12
    diff_threshold: int = 8
    edge_low: int = 15
    edge_high: int = 90
    bg_open_size: int = 7
    bg_close_size: int = 9
    bg_gate_dilate: int = 15
    solidity_min: float = 0.30
    solidity_min_far: float = 0.22
    extent_min: float = 0.22
    extent_min_far: float = 0.12
    shape_area_padding: int = 3
    shape_score_threshold: float = 0.45
    shape_score_threshold_far: float = 0.60
    shape_score_gap: float = 0.01
    shape_score_gap_lt: float = 0.03
    far_area_ratio_threshold: float = 0.0012
    color_black_v: int = 35
    color_low_sat: int = 45
    fg_threshold: int = 80
    fg_alpha: float = 0.35
    fg_decay: float = 0.82
    fg_init_frames: int = 10
    defect_depth_threshold: float = 4.0
    defect_penalty: float = 0.09
    side_count_penalty: float = 0.035
    motion_px_for_stale_bg: int = 4500
    no_contour_reset_limit: int = 12
    template_canvas_size: int = 96
    shape_switch_frames: int = 6
    shape_switch_frames_unknown: int = 10
    max_candidate_aspect_ratio: float = 4.0
    min_candidate_short_side_px: int = 12
    min_candidate_sides: int = 5
    min_candidate_compactness: float = 0.012
    repair_close_size: int = 7
    contour_hold_frames: int = 4


class TemplateClassifier:
    def __init__(self, config: DetectorConfig) -> None:
        self.cfg = config
        self.template_bank = self._build_template_bank()
        self.template_bitmaps: Dict[str, List[np.ndarray]] = {}
        self.template_defect_counts: Dict[str, int] = {}
        self.template_defect_profiles: Dict[str, List[Tuple[int, float, float]]] = {}
        self.template_side_counts: Dict[str, int] = {}

        for label, templates in self.template_bank.items():
            bitmaps: List[np.ndarray] = []
            defect_counts: List[int] = []
            profiles: List[Tuple[int, float, float]] = []
            side_counts: List[int] = []
            for template in templates:
                bitmaps.append(self._contour_to_canvas(template, self.cfg.template_canvas_size))
                defect_counts.append(self._convex_defect_count(template))
                profiles.append(self._convex_defect_profile(template))
                perim = cv2.arcLength(template, True)
                eps = 0.030 * perim if perim > 0 else 0.0
                poly = cv2.approxPolyDP(template, eps, True) if eps > 0 else template
                side_counts.append(len(poly))

            self.template_bitmaps[label] = bitmaps
            self.template_defect_counts[label] = int(round(np.median(defect_counts))) if defect_counts else 0
            self.template_defect_profiles[label] = profiles
            self.template_side_counts[label] = int(round(np.median(side_counts))) if side_counts else 0

    def classify_shape(self, contour: np.ndarray, score_threshold: float, debug: List[str]) -> str:
        best_label = "Unknown"
        best_score = 1e9
        second_score = 1e9
        best_label_raw = 1e9
        second_label_raw = 1e9
        second_label = "Unknown"

        if contour is None or len(contour) < 3:
            return best_label

        epsilon = 0.030 * cv2.arcLength(contour, True)
        normalized = cv2.approxPolyDP(contour, epsilon, True)
        candidate_side_count = len(normalized)

        candidate_bitmap = self._contour_to_canvas(normalized, self.cfg.template_canvas_size)
        if candidate_bitmap is None:
            candidate_bitmap = np.zeros((self.cfg.template_canvas_size, self.cfg.template_canvas_size), dtype=np.uint8)

        candidate_profile = self._convex_defect_profile(normalized)
        candidate_defect_count = self._convex_defect_count(normalized)

        for label, templates in self.template_bank.items():
            label_best_score = 1e9
            for idx, template in enumerate(templates):
                match_score = cv2.matchShapes(normalized, template, cv2.CONTOURS_MATCH_I1, 0.0)
                tb = self.template_bitmaps.get(label, [])[idx]
                if tb is not None:
                    diff = cv2.countNonZero(cv2.absdiff(candidate_bitmap, tb))
                    raster_score = diff / float(self.cfg.template_canvas_size * self.cfg.template_canvas_size)
                    match_score = min(match_score, raster_score)
                label_best_score = min(label_best_score, match_score)

            geom_delta = abs(candidate_defect_count - self.template_defect_counts.get(label, 0)) * self.cfg.defect_penalty
            side_delta = abs(candidate_side_count - self.template_side_counts.get(label, candidate_side_count)) * self.cfg.side_count_penalty
            combined = label_best_score + geom_delta + side_delta
            debug.append(
                f"classify:{label} best={label_best_score:.3f} geom={geom_delta:.3f} sides={side_delta:.3f} combined={combined:.3f}"
            )

            if combined < second_score:
                if combined < best_score:
                    second_score = best_score
                    second_label = best_label
                    second_label_raw = best_label_raw
                    best_score = combined
                    best_label_raw = label_best_score
                    best_label = label
                else:
                    second_score = combined
                    second_label = label
                    second_label_raw = label_best_score

        if best_score > score_threshold:
            return "Unknown"

        def closest_profile(name: str) -> Tuple[int, float, float]:
            profiles = self.template_defect_profiles.get(name, [])
            if not profiles:
                return (0, 0.0, 0.0)
            return min(profiles, key=lambda p: abs(candidate_defect_count - p[0]) + abs(candidate_profile[0] - p[0]))

        def profile_delta(profile: Tuple[int, float, float]) -> float:
            return (
                abs(candidate_profile[0] - profile[0])
                + abs(candidate_profile[1] - profile[1])
                + abs(candidate_profile[2] - profile[2])
            )

        if second_label in ("L", "Skew") and best_label in ("L", "Skew") and (second_score - best_score) < self.cfg.shape_score_gap:
            bprof = closest_profile(best_label)
            sprof = closest_profile(second_label)
            if profile_delta(sprof) < profile_delta(bprof):
                best_label = second_label
                best_score, second_score = second_score, best_score
                best_label_raw, second_label_raw = second_label_raw, best_label_raw
                debug.append(f"classify tie-break: {best_label} closer in defect count")

        if second_label in ("L", "T") and best_label in ("L", "T") and (second_score - best_score) < self.cfg.shape_score_gap_lt:
            bprof = closest_profile(best_label)
            sprof = closest_profile(second_label)
            if profile_delta(sprof) < profile_delta(bprof):
                best_label = second_label
                best_score, second_score = second_score, best_score
                best_label_raw, second_label_raw = second_label_raw, best_label_raw
                debug.append(f"classify tie-break LT: {best_label} closer in defect profile")

        if second_label_raw < 1e9:
            debug.append(f"classify raw scores: best={best_label_raw:.3f} second={second_label_raw:.3f} gap={second_label_raw - best_label_raw:.3f}")

        return f"{best_label} ({best_score:.3f})"

    def classify_color(self, frame: np.ndarray, contour: np.ndarray) -> str:
        if contour is None or len(contour) < 3:
            return "Unknown"

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        masked = cv2.bitwise_and(frame, frame, mask=mask)
        hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)

        v = hsv[:, :, 2][mask > 0]
        if v.size == 0:
            return "Unknown"
        if np.mean(v) <= self.cfg.color_black_v:
            return "Black"

        s = hsv[:, :, 1][mask > 0]
        h = hsv[:, :, 0][mask > 0]
        valid = s >= self.cfg.color_low_sat
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

    def _build_template_bank(self) -> Dict[str, List[np.ndarray]]:
        t = self._make_template_contour([(1, 0), (0, 1), (1, 1), (2, 1)])
        skew = self._make_template_contour([(0, 0), (1, 0), (1, 1), (2, 1)])
        l = self._make_template_contour([(0, 0), (0, 1), (1, 1), (2, 1)])
        return {
            "T": [t] + self._rotations(t),
            "Skew": [skew] + self._rotations(skew),
            "L": [l] + self._rotations(l),
        }

    def _make_template_contour(self, blocks: List[Tuple[int, int]]) -> np.ndarray:
        canvas = np.zeros((200, 200), dtype=np.uint8)
        cell = 24
        offset = 20
        for bx, by in blocks:
            x0 = offset + bx * cell
            y0 = offset + by * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell, y0 + cell), 255, -1)
        contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours[0]

    def _rotations(self, contour: np.ndarray) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        bbox = cv2.boundingRect(contour)
        center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
        for angle in (0, 90, 180, 270):
            m = cv2.getRotationMatrix2D(center, angle, 1.0)
            out.append(cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), m).astype(np.float32))
        return out

    def _contour_to_canvas(self, contour: np.ndarray, size: int) -> Optional[np.ndarray]:
        if contour is None or len(contour) < 3:
            return None
        points = contour.astype(np.float32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(points.astype(np.int32))
        if w <= 0 or h <= 0:
            return None
        usable = max(1, size - 8)
        scale = usable / float(max(w, h))
        shift_x = 4 - x * scale + max(0.0, (usable - w * scale) / 2.0)
        shift_y = 4 - y * scale + max(0.0, (usable - h * scale) / 2.0)
        mat = np.array([[scale, 0.0, shift_x], [0.0, scale, shift_y]], dtype=np.float32)
        norm = cv2.transform(points, mat).reshape(-1, 1, 2)
        canvas = np.zeros((size, size), dtype=np.uint8)
        cv2.drawContours(canvas, [norm.astype(np.int32)], -1, 255, -1)
        cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), dst=canvas)
        return canvas

    def _convex_defect_count(self, contour: np.ndarray) -> int:
        profile = self._convex_defect_profile(contour)
        return profile[0]

    def _convex_defect_profile(self, contour: np.ndarray) -> Tuple[int, float, float]:
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

        depths = [
            float(d[0][3]) / 256.0
            for d in defects
            if float(d[0][3]) / 256.0 >= self.cfg.defect_depth_threshold * 0.25
        ]
        if not depths:
            return (0, 0.0, 0.0)
        arr = np.array(depths, dtype=np.float32)
        return (int(len(arr)), float(np.mean(arr)), float(np.max(arr)))


class ShapeStabilizer:
    def __init__(self, switch_frames: int, switch_unknown_frames: int) -> None:
        self.switch_frames = switch_frames
        self.switch_unknown_frames = switch_unknown_frames
        self.stable: Optional[str] = None
        self.pending: Optional[str] = None
        self.pending_count = 0

    def reset(self) -> None:
        self.stable = None
        self.pending = None
        self.pending_count = 0

    def update(self, candidate: str) -> str:
        if not candidate:
            candidate = "Unknown"

        if self.stable is None:
            self.stable = candidate
            return self.stable

        if candidate == self.stable:
            self.pending = None
            self.pending_count = 0
            return self.stable

        if candidate == self.pending:
            self.pending_count += 1
        else:
            self.pending = candidate
            self.pending_count = 1

        needed = self.switch_unknown_frames if candidate == "Unknown" else self.switch_frames
        if self.pending_count >= needed:
            self.stable = candidate
            self.pending = None
            self.pending_count = 0

        return self.stable


class PartSorterDetector:
    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.cfg = config or DetectorConfig()
        self.classifier = TemplateClassifier(self.cfg)
        self.stabilizer = ShapeStabilizer(self.cfg.shape_switch_frames, self.cfg.shape_switch_frames_unknown)

        self.background_frame: Optional[np.ndarray] = None
        self.background_calibrated = False
        self.calibration_frames = 0
        self.fg_subtractor = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=50, detectShadows=False)
        self.fg_stable_mask: Optional[np.ndarray] = None
        self.no_contour_streak = 0
        self.last_detection_key: Optional[str] = None
        self.last_action: Optional[str] = None
        self.last_visual_contour: Optional[np.ndarray] = None
        self.last_visual_label: Optional[str] = None
        self.last_visual_color: Optional[str] = None
        self.held_frames = 0

    def reset_motion_models(self) -> None:
        self.background_frame = None
        self.background_calibrated = False
        self.calibration_frames = 0
        self.fg_stable_mask = None
        self.fg_subtractor = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=50, detectShadows=False)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
        working = frame.copy()
        frame_area = working.shape[0] * working.shape[1]

        near_min_area = max(self.cfg.min_shape_area, int(frame_area * self.cfg.min_shape_area_ratio_near))
        far_min_area = max(self.cfg.min_shape_area_far, int(frame_area * self.cfg.min_shape_area_ratio_far))
        debug = [f"frame={frame.shape[1]}x{frame.shape[0]}", f"calibrated={self.background_calibrated}"]

        if not self.background_calibrated:
            if self.background_frame is None:
                self.background_frame = np.float32(frame)
            else:
                self.background_frame += frame
            self.calibration_frames += 1
            if self.calibration_frames >= self.cfg.calibration_frame_count:
                self.background_frame = np.clip(
                    self.background_frame / float(self.calibration_frames), 0, 255
                ).astype(np.uint8)
                self.background_calibrated = True

        if self.background_calibrated and self.background_frame is not None:
            debug.append("bg_model: yes")
            diff = cv2.absdiff(
                cv2.cvtColor(working, cv2.COLOR_BGR2LAB),
                cv2.cvtColor(self.background_frame, cv2.COLOR_BGR2LAB),
            )
            diff_mag = np.sqrt(np.sum(diff.astype(np.float32) ** 2, axis=2))
            _, bg_mask = cv2.threshold(diff_mag.astype(np.uint8), self.cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        else:
            debug.append("bg_model: no")
            bg_mask = np.zeros((working.shape[0], working.shape[1]), dtype=np.uint8)

        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, np.ones((self.cfg.bg_open_size, self.cfg.bg_open_size), np.uint8))
        bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, np.ones((self.cfg.bg_close_size, self.cfg.bg_close_size), np.uint8))

        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        lr = 0.01 if self.calibration_frames < self.cfg.fg_init_frames else 0.0
        _, moving_raw = cv2.threshold(
            self.fg_subtractor.apply(blurred, learningRate=lr),
            self.cfg.fg_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        moving_raw = cv2.morphologyEx(moving_raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        moving_raw = cv2.morphologyEx(moving_raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        raw_float = moving_raw.astype(np.float32) / 255.0
        if self.fg_stable_mask is None:
            self.fg_stable_mask = raw_float.copy()
        else:
            self.fg_stable_mask = self.fg_stable_mask * self.cfg.fg_decay
            self.fg_stable_mask = cv2.addWeighted(raw_float, self.cfg.fg_alpha, self.fg_stable_mask, 1.0 - self.cfg.fg_alpha, 0.0)

        _, stable_move = cv2.threshold(self.fg_stable_mask, 0.50, 1.0, cv2.THRESH_BINARY)
        moving_mask = (stable_move * 255).astype(np.uint8)

        edges = cv2.Canny(blurred, self.cfg.edge_low, self.cfg.edge_high)
        edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        bg_gate = cv2.dilate(bg_mask, np.ones((self.cfg.bg_gate_dilate, self.cfg.bg_gate_dilate), np.uint8), iterations=1)
        moving_pixels = int(np.count_nonzero(moving_mask))

        low_motion = moving_pixels < self.cfg.motion_px_for_stale_bg
        if low_motion:
            shape_mask = cv2.bitwise_or(bg_mask, cv2.bitwise_and(edge_mask, bg_gate))
        else:
            shape_mask = cv2.bitwise_and(edge_mask, bg_gate)

        foreground_gate = moving_mask
        if self.background_calibrated and not low_motion:
            foreground_gate = cv2.bitwise_or(foreground_gate, bg_mask)

        shape_mask = cv2.bitwise_and(
            shape_mask,
            cv2.morphologyEx(foreground_gate, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8)),
        )
        shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        shape_mask = cv2.medianBlur(shape_mask, 5)
        _, shape_mask = cv2.threshold(shape_mask, 127, 255, cv2.THRESH_BINARY)
        shape_mask = cv2.morphologyEx(
            shape_mask,
            cv2.MORPH_CLOSE,
            np.ones((self.cfg.repair_close_size, self.cfg.repair_close_size), np.uint8),
        )

        debug_mask = cv2.cvtColor(shape_mask, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            debug_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

        if not contours:
            self.no_contour_streak += 1
            if self.no_contour_streak >= self.cfg.no_contour_reset_limit:
                self.reset_motion_models()
                self.no_contour_streak = 0
            if self._draw_held_visual(working, debug):
                return working, debug_mask, debug, None
            self._clear_visual_state()
            return working, debug_mask, debug, None

        self.no_contour_streak = 0
        max_area = frame_area * self.cfg.max_shape_area_ratio
        candidates: List[Tuple[float, np.ndarray]] = []

        for c in contours:
            area = cv2.contourArea(c)
            ratio = area / float(frame_area) if frame_area else 0.0
            is_far = ratio <= self.cfg.far_area_ratio_threshold
            min_area = far_min_area if is_far else near_min_area
            min_sol = self.cfg.solidity_min_far if is_far else self.cfg.solidity_min
            min_ext = self.cfg.extent_min_far if is_far else self.cfg.extent_min
            if area <= min_area or area > max_area:
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            short_side = min(w, h)
            long_side = max(w, h)
            if short_side < self.cfg.min_candidate_short_side_px:
                continue
            if (long_side / float(short_side)) > self.cfg.max_candidate_aspect_ratio:
                continue
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            compactness = area / float(perim * perim)
            if compactness < self.cfg.min_candidate_compactness:
                continue
            approx = cv2.approxPolyDP(c, 0.03 * perim, True)
            if len(approx) < self.cfg.min_candidate_sides:
                continue
            if (area / hull_area) < min_sol or (area / float(w * h)) < min_ext:
                continue
            candidates.append((area, c))

        if not candidates:
            debug.append("no valid contour candidates")
            if self._draw_held_visual(working, debug):
                return working, debug_mask, debug, None
            self._clear_visual_state()
            return working, debug_mask, debug, None

        largest = max(candidates, key=lambda t: t[0])[1]
        area = cv2.contourArea(largest)
        x, y, w, h = cv2.boundingRect(largest)

        area_ratio = area / float(frame_area) if frame_area else 0.0
        threshold = self.cfg.shape_score_threshold_far if area_ratio <= self.cfg.far_area_ratio_threshold else self.cfg.shape_score_threshold
        raw_shape_label = self.classifier.classify_shape(largest, threshold, debug)
        raw_name = self._clean_shape_name(raw_shape_label)
        raw_score = self._extract_score(raw_shape_label)
        stable_name = self.stabilizer.update(raw_name)
        if stable_name == "Unknown":
            debug.append("stable label unknown; skipping draw/action")
            if self._draw_held_visual(working, debug):
                return working, debug_mask, debug, None
            self._clear_visual_state()
            return working, debug_mask, debug, None

        cv2.drawContours(working, [largest], -1, (0, 255, 0), 2)
        color_label = self.classifier.classify_color(working, largest)
        self.last_visual_contour = largest.copy()
        self.last_visual_label = stable_name
        self.last_visual_color = color_label
        self.held_frames = 0

        if raw_score is None:
            label = f"{stable_name} / {color_label}"
        else:
            label = f"{stable_name} ({raw_score:.3f}) / {color_label}"

        cv2.putText(
            working,
            label,
            (x + 6, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        action = self._direction(stable_name, color_label)
        key = f"{stable_name}|{color_label}"
        should_emit = self.last_detection_key != key or self.last_action != action
        if should_emit:
            self.last_detection_key = key
            self.last_action = action
        else:
            action = None

        debug.append(f"label={label}")
        return working, debug_mask, debug, action

    def _draw_held_visual(self, working: np.ndarray, debug: List[str]) -> bool:
        if self.last_visual_contour is None:
            return False
        if self.held_frames >= self.cfg.contour_hold_frames:
            self._clear_visual_state()
            return False
        self.held_frames += 1
        cv2.drawContours(working, [self.last_visual_contour], -1, (0, 255, 0), 2)
        x, y, _, _ = cv2.boundingRect(self.last_visual_contour)
        label = f"{self.last_visual_label} / {self.last_visual_color}"
        cv2.putText(
            working,
            label,
            (x + 6, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        debug.append(f"holding previous contour frame={self.held_frames}")
        return True

    def _clear_visual_state(self) -> None:
        self.last_visual_contour = None
        self.last_visual_label = None
        self.last_visual_color = None
        self.held_frames = 0
        self.stabilizer.reset()
        self.last_action = None
        self.last_detection_key = None

    @staticmethod
    def _clean_shape_name(shape_with_score: str) -> str:
        return shape_with_score.split("(")[0].strip() if shape_with_score else "Unknown"

    @staticmethod
    def _extract_score(shape_with_score: str) -> Optional[float]:
        if not shape_with_score:
            return None
        start = shape_with_score.find("(")
        end = shape_with_score.find(")", start + 1)
        if start == -1 or end == -1:
            return None
        try:
            return float(shape_with_score[start + 1 : end])
        except Exception:
            return None

    @staticmethod
    def _direction(shape_name: str, color_name: str) -> str:
        if shape_name == "Skew" and color_name == "Orange":
            return "left"
        if shape_name == "L" and color_name == "Green":
            return "left"
        if shape_name == "Skew" and color_name == "Green":
            return "right"
        if shape_name == "L" and color_name == "Orange":
            return "right"
        return "outward"
