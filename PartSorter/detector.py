from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from shape_classifier import ShapeClassifier, classify_color, clean_shape_name, direction, extract_score


MIN_SHAPE_AREA = 120
MIN_SHAPE_AREA_FAR = 40
MIN_SHAPE_AREA_RATIO_NEAR = 0.0002
MIN_SHAPE_AREA_RATIO_FAR = 0.00008
MAX_SHAPE_AREA_RATIO = 0.30
FAR_AREA_RATIO_THRESHOLD = 0.0012

CALIBRATION_FRAME_COUNT = 12
DIFF_THRESHOLD = 8
EDGE_LOW = 15
EDGE_HIGH = 90
FG_THRESHOLD = 80
FG_ALPHA = 0.35
FG_DECAY = 0.82
FG_INIT_FRAMES = 10
MOTION_PX_FOR_STALE_BG = 4500
NO_CONTOUR_RESET_LIMIT = 12

BG_OPEN_SIZE = 7
BG_CLOSE_SIZE = 9
BG_GATE_DILATE = 15
REPAIR_CLOSE_SIZE = 7

SOLIDITY_MIN = 0.30
SOLIDITY_MIN_FAR = 0.22
EXTENT_MIN = 0.22
EXTENT_MIN_FAR = 0.12

MAX_CANDIDATE_ASPECT_RATIO = 4.0
MIN_CANDIDATE_SHORT_SIDE_PX = 12
MIN_CANDIDATE_SIDES = 5
MIN_CANDIDATE_COMPACTNESS = 0.012

SHAPE_SCORE_THRESHOLD = 0.45
SHAPE_SCORE_THRESHOLD_FAR = 0.60
SHAPE_SWITCH_FRAMES = 6
SHAPE_SWITCH_FRAMES_UNKNOWN = 10
CONTOUR_HOLD_FRAMES = 4


class PartSorterDetector:
    def __init__(self) -> None:
        # Runtime-tunable area thresholds used by the Tk slider.
        self.min_shape_area = MIN_SHAPE_AREA
        self.min_shape_area_far = MIN_SHAPE_AREA_FAR

        # Background / motion model state.
        self.background_frame: Optional[np.ndarray] = None
        self.background_calibrated = False
        self.calibration_frames = 0
        self.fg_subtractor = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=50, detectShadows=False)
        self.fg_stable_mask: Optional[np.ndarray] = None

        # Command emission de-duplication.
        self.no_contour_streak = 0
        self.last_detection_key: Optional[str] = None
        self.last_action: Optional[str] = None

        # Temporal label stabilization state.
        self.stable_name: Optional[str] = None
        self.pending_name: Optional[str] = None
        self.pending_count = 0

        # Short visual hold to avoid outline drops on single bad frames.
        self.last_visual_contour: Optional[np.ndarray] = None
        self.last_visual_label: Optional[str] = None
        self.last_visual_color: Optional[str] = None
        self.held_frames = 0

        self.classifier = ShapeClassifier()

    def set_min_object_area(self, min_area: int) -> None:
        self.min_shape_area = max(1, int(min_area))
        self.min_shape_area_far = max(10, self.min_shape_area // 3)

    def reset_motion_models(self) -> None:
        self.background_frame = None
        self.background_calibrated = False
        self.calibration_frames = 0
        self.fg_stable_mask = None
        self.fg_subtractor = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=50, detectShadows=False)

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[str]]:
        # Stage 1: build masks (background change + motion + edges).
        working = frame.copy()
        frame_area = working.shape[0] * working.shape[1]
        near_min_area = max(self.min_shape_area, int(frame_area * MIN_SHAPE_AREA_RATIO_NEAR))
        far_min_area = max(self.min_shape_area_far, int(frame_area * MIN_SHAPE_AREA_RATIO_FAR))
        debug = [f"frame={frame.shape[1]}x{frame.shape[0]}", f"calibrated={self.background_calibrated}"]

        bg_mask = self._build_background_mask(working, debug)
        shape_mask, edge_mask = self._build_shape_mask(working, bg_mask)
        debug_mask = cv2.cvtColor(shape_mask, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            # Fallback to pure edge contours when the fused mask is empty.
            contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            debug_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

        if not contours:
            return self._missing(working, debug_mask, debug)

        contour = self._pick_candidate(contours, frame_area, near_min_area, far_min_area)
        if contour is None:
            debug.append("no valid contour candidates")
            return self._missing(working, debug_mask, debug)

        # Stage 2: classify shape and color for the best candidate.
        self.no_contour_streak = 0
        area_ratio = cv2.contourArea(contour) / float(frame_area) if frame_area else 0.0
        threshold = SHAPE_SCORE_THRESHOLD_FAR if area_ratio <= FAR_AREA_RATIO_THRESHOLD else SHAPE_SCORE_THRESHOLD

        raw_shape = self.classifier.classify_shape(contour, threshold, debug)
        stable_name = self._stabilize(clean_shape_name(raw_shape))
        if stable_name == "Unknown":
            debug.append("stable label unknown; skipping draw/action")
            if self._draw_held(working, debug):
                return working, debug_mask, debug, None
            self._clear_state()
            return working, debug_mask, debug, None

        raw_score = extract_score(raw_shape)
        color = classify_color(working, contour)
        x, y, _, _ = cv2.boundingRect(contour)
        label = f"{stable_name} / {color}" if raw_score is None else f"{stable_name} ({raw_score:.3f}) / {color}"

        cv2.drawContours(working, [contour], -1, (0, 255, 0), 2)
        cv2.putText(working, label, (x + 6, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        self.last_visual_contour = contour.copy()
        self.last_visual_label = stable_name
        self.last_visual_color = color
        self.held_frames = 0

        # Only emit a command when the logical detection actually changes.
        action = direction(stable_name, color)
        key = f"{stable_name}|{color}"
        if self.last_detection_key == key and self.last_action == action:
            action = None
        else:
            self.last_detection_key = key
            self.last_action = action

        debug.append(f"label={label}")
        return working, debug_mask, debug, action

    def _missing(self, working: np.ndarray, debug_mask: np.ndarray, debug: List[str]):
        # Missing candidate can be transient; hold previous contour briefly.
        self.no_contour_streak += 1
        if self.no_contour_streak >= NO_CONTOUR_RESET_LIMIT:
            self.reset_motion_models()
            self.no_contour_streak = 0
        if self._draw_held(working, debug):
            return working, debug_mask, debug, None
        self._clear_state()
        return working, debug_mask, debug, None

    def _build_background_mask(self, frame: np.ndarray, debug: List[str]) -> np.ndarray:
        # Bootstrap background from the first N frames, then compare in LAB space.
        if not self.background_calibrated:
            if self.background_frame is None:
                self.background_frame = np.float32(frame)
            else:
                self.background_frame += frame
            self.calibration_frames += 1
            if self.calibration_frames >= CALIBRATION_FRAME_COUNT:
                self.background_frame = np.clip(self.background_frame / float(self.calibration_frames), 0, 255).astype(np.uint8)
                self.background_calibrated = True

        if self.background_calibrated and self.background_frame is not None:
            debug.append("bg_model: yes")
            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), cv2.cvtColor(self.background_frame, cv2.COLOR_BGR2LAB))
            diff_mag = np.sqrt(np.sum(diff.astype(np.float32) ** 2, axis=2))
            _, mask = cv2.threshold(diff_mag.astype(np.uint8), DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        else:
            debug.append("bg_model: no")
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((BG_OPEN_SIZE, BG_OPEN_SIZE), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((BG_CLOSE_SIZE, BG_CLOSE_SIZE), np.uint8))
        return mask

    def _build_shape_mask(self, frame: np.ndarray, bg_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Fuse background-diff mask, motion mask, and edges into one candidate mask.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        lr = 0.01 if self.calibration_frames < FG_INIT_FRAMES else 0.0
        _, moving = cv2.threshold(self.fg_subtractor.apply(blurred, learningRate=lr), FG_THRESHOLD, 255, cv2.THRESH_BINARY)
        moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        moving = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        rawf = moving.astype(np.float32) / 255.0
        if self.fg_stable_mask is None:
            self.fg_stable_mask = rawf.copy()
        else:
            self.fg_stable_mask = self.fg_stable_mask * FG_DECAY
            self.fg_stable_mask = cv2.addWeighted(rawf, FG_ALPHA, self.fg_stable_mask, 1.0 - FG_ALPHA, 0.0)
        _, stable = cv2.threshold(self.fg_stable_mask, 0.50, 1.0, cv2.THRESH_BINARY)
        moving_mask = (stable * 255).astype(np.uint8)

        edges = cv2.Canny(blurred, EDGE_LOW, EDGE_HIGH)
        edge_mask = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
        bg_gate = cv2.dilate(bg_mask, np.ones((BG_GATE_DILATE, BG_GATE_DILATE), np.uint8), iterations=1)

        low_motion = int(np.count_nonzero(moving_mask)) < MOTION_PX_FOR_STALE_BG
        shape_mask = cv2.bitwise_or(bg_mask, cv2.bitwise_and(edge_mask, bg_gate)) if low_motion else cv2.bitwise_and(edge_mask, bg_gate)

        fg_gate = moving_mask if (not self.background_calibrated or low_motion) else cv2.bitwise_or(moving_mask, bg_mask)
        shape_mask = cv2.bitwise_and(shape_mask, cv2.morphologyEx(fg_gate, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8)))

        shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        shape_mask = cv2.medianBlur(shape_mask, 5)
        _, shape_mask = cv2.threshold(shape_mask, 127, 255, cv2.THRESH_BINARY)
        shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, np.ones((REPAIR_CLOSE_SIZE, REPAIR_CLOSE_SIZE), np.uint8))
        return shape_mask, edge_mask

    def _pick_candidate(self, contours: List[np.ndarray], frame_area: int, near_min_area: int, far_min_area: int) -> Optional[np.ndarray]:
        # Always prefer the largest object: sort by area descending, then accept
        # the first contour that passes basic sanity checks.
        max_area = frame_area * MAX_SHAPE_AREA_RATIO
        by_area = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in by_area:
            area = cv2.contourArea(c)
            if area <= 0 or area > max_area:
                continue

            ratio = area / float(frame_area) if frame_area else 0.0
            is_far = ratio <= FAR_AREA_RATIO_THRESHOLD
            min_area = far_min_area if is_far else near_min_area
            min_sol = SOLIDITY_MIN_FAR if is_far else SOLIDITY_MIN
            min_ext = EXTENT_MIN_FAR if is_far else EXTENT_MIN
            if area <= min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w <= 0 or h <= 0:
                continue
            short_side, long_side = min(w, h), max(w, h)
            if short_side < MIN_CANDIDATE_SHORT_SIDE_PX or (long_side / float(short_side)) > MAX_CANDIDATE_ASPECT_RATIO:
                continue

            # Keep minimum anti-line checks, but do not rank by these metrics.
            perim = cv2.arcLength(c, True)
            if perim <= 0 or (area / float(perim * perim)) < MIN_CANDIDATE_COMPACTNESS:
                continue
            if len(cv2.approxPolyDP(c, 0.03 * perim, True)) < MIN_CANDIDATE_SIDES:
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            if (area / hull_area) < min_sol or (area / float(w * h)) < min_ext:
                continue

            return c

        return None

    def _stabilize(self, candidate: str) -> str:
        # Hysteresis: require repeated frames before switching labels.
        candidate = candidate or "Unknown"
        if self.stable_name is None:
            self.stable_name = candidate
            return self.stable_name
        if candidate == self.stable_name:
            self.pending_name = None
            self.pending_count = 0
            return self.stable_name

        if candidate == self.pending_name:
            self.pending_count += 1
        else:
            self.pending_name = candidate
            self.pending_count = 1

        needed = SHAPE_SWITCH_FRAMES_UNKNOWN if candidate == "Unknown" else SHAPE_SWITCH_FRAMES
        if self.pending_count >= needed:
            self.stable_name = candidate
            self.pending_name = None
            self.pending_count = 0
        return self.stable_name

    def _draw_held(self, frame: np.ndarray, debug: List[str]) -> bool:
        # Keeps outline stable when one corner/edge is briefly missed.
        if self.last_visual_contour is None or self.held_frames >= CONTOUR_HOLD_FRAMES:
            return False
        self.held_frames += 1
        cv2.drawContours(frame, [self.last_visual_contour], -1, (0, 255, 0), 2)
        x, y, _, _ = cv2.boundingRect(self.last_visual_contour)
        cv2.putText(frame, f"{self.last_visual_label} / {self.last_visual_color}", (x + 6, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        debug.append(f"holding previous contour frame={self.held_frames}")
        return True

    def _clear_state(self) -> None:
        self.last_visual_contour = None
        self.last_visual_label = None
        self.last_visual_color = None
        self.held_frames = 0
        self.stable_name = None
        self.pending_name = None
        self.pending_count = 0
        self.last_action = None
        self.last_detection_key = None
        
