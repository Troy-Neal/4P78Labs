import time
import sys
import os
#import nxt
#import nxt.locator
#import nxt.motor
#import nxt.sensor
#import nxt.sensor.generic
import cv2
import numpy as np
import tkinter as tk
import base64

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
background_frame = None
background_calibrated = False
calibration_frames = 0
running = True
app_open = True
MIN_SHAPE_AREA = 120
MAX_SHAPE_AREA_RATIO = 0.30
MIN_SHAPE_AREA_FAR = 40
MIN_SHAPE_AREA_RATIO_NEAR = 0.0002
MIN_SHAPE_AREA_RATIO_FAR = 0.00008
CALIBRATION_FRAME_COUNT = 12
DIFF_THRESHOLD = 8
EDGE_LOW = 15
EDGE_HIGH = 90
TARGET_FPS = 8
PROCESS_INTERVAL_MS = int(1000 / TARGET_FPS)
BG_OPEN_SIZE = 7
BG_CLOSE_SIZE = 9
BG_GATE_DILATE = 15
SOLIDITY_MIN = 0.30
SOLIDITY_MIN_FAR = 0.22
EXTENT_MIN = 0.22
EXTENT_MIN_FAR = 0.12
SHAPE_AREA_PADDING = 3
SHAPE_SCORE_THRESHOLD = 0.45
SHAPE_SCORE_THRESHOLD_FAR = 0.60
SHAPE_SCORE_GAP = 0.01
FAR_AREA_RATIO_THRESHOLD = 0.0012
L_SKEW_PROFILE_SWAP_MARGIN = 0.08
COLOR_BLACK_V = 35
COLOR_LOW_SAT = 45
COLOR_BLACK = "Black"
COLOR_WHITE = "White"
COLOR_GRAY = "Gray"
FG_THRESHOLD = 80
FG_ALPHA = 0.35
FG_DECAY = 0.82
FG_INIT_FRAMES = 10
DEFECT_DEPTH_THRESHOLD = 4.0
DEFECT_PENALTY = 0.09
DEFECT_MEAN_PENALTY = 0.03
PROFILE_PENALTY_MAX = 0.35
DEBUG_INFO = []
DEBUG_TICK = 0
SHOW_DEBUG_TEXT = False
DEBUG_MASK = None
FG_SUBTRACTOR = cv2.createBackgroundSubtractorMOG2(history=600, varThreshold=50, detectShadows=False)
FG_STABLE_MASK = None
TEMPLATE_CANVAS_SIZE = 96
DEBUG_PRINT_EVERY = 30
DEBUG_PRINT_LINES = 22
DEBUG_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detection_debug.log")
DEBUG_LOG_FILE = None


def _open_debug_log():
	global DEBUG_LOG_FILE
	if DEBUG_LOG_FILE is not None:
		return
	try:
		DEBUG_LOG_FILE = open(DEBUG_LOG_PATH, "a", encoding="utf-8")
	except Exception:
		DEBUG_LOG_FILE = None


def _log_debug(lines):
	_open_debug_log()
	if DEBUG_LOG_FILE is None:
		return
	try:
		timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
		DEBUG_LOG_FILE.write(f"[{timestamp}] {lines}\n")
		DEBUG_LOG_FILE.flush()
	except Exception:
		pass


def _set_tk_image(target_label, frame):
	if frame is None or target_label is None:
		return
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	encoded, buffer = cv2.imencode(".png", rgb)
	if not encoded:
		return
	photo = tk.PhotoImage(data=base64.b64encode(buffer).decode("ascii"), format="png")
	target_label.config(image=photo)
	target_label.image = photo


def _make_template_contour(blocks):
	canvas_size = 200
	cell = 24
	offset = 20
	canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
	for bx, by in blocks:
		x0 = offset + bx * cell
		y0 = offset + by * cell
		x1 = x0 + cell
		y1 = y0 + cell
		cv2.rectangle(canvas, (x0, y0), (x1, y1), 255, -1)
	contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if not contours:
		return None
	return contours[0]


def _rotations(contour):
	rotated = []
	for angle in (0, 90, 180, 270):
		M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
		bbox = cv2.boundingRect(contour)
		center = (bbox[0] + bbox[2] / 2.0, bbox[1] + bbox[3] / 2.0)
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		r = cv2.transform(contour.reshape(-1, 1, 2).astype(np.float32), M)
		rotated.append(r.astype(np.float32))
	return rotated


_T_TEMPLATE = _make_template_contour([(1, 0), (0, 1), (1, 1), (2, 1)])
_SKew_TEMPLATE = _make_template_contour([(0, 0), (1, 0), (1, 1), (2, 1)])
_L_TEMPLATE = _make_template_contour([(0, 0), (0, 1), (1, 1), (2, 1)])
def _contour_to_canvas(contour, size=TEMPLATE_CANVAS_SIZE):
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
	M = np.array([[scale, 0.0, shift_x], [0.0, scale, shift_y]], dtype=np.float32)
	norm = cv2.transform(points, M).reshape(-1, 1, 2)
	canvas = np.zeros((size, size), dtype=np.uint8)
	cv2.drawContours(canvas, [norm.astype(np.int32)], -1, 255, -1)
	cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), dst=canvas)
	return canvas


def _convex_defect_count(contour):
	if contour is None or len(contour) < 4:
		return 0
	c = contour.astype(np.int32)
	try:
		hull = cv2.convexHull(c, returnPoints=False)
	except Exception:
		return 0
	if hull is None or len(hull) < 3:
		return 0
	try:
		defects = cv2.convexityDefects(c, hull)
	except Exception:
		return 0
	if defects is None:
		return 0
	count = 0
	for defect in defects:
		depth = float(defect[0][3]) / 256.0
		if depth >= DEFECT_DEPTH_THRESHOLD:
			count += 1
	return count


def _convex_defect_profile(contour):
	if contour is None or len(contour) < 4:
		return (0, 0.0, 0.0)
	c = contour.astype(np.int32)
	try:
		hull = cv2.convexHull(c, returnPoints=False)
	except Exception:
		return (0, 0.0, 0.0)
	if hull is None or len(hull) < 3:
		return (0, 0.0, 0.0)
	try:
		defects = cv2.convexityDefects(c, hull)
	except Exception:
		return (0, 0.0, 0.0)
	if defects is None or len(defects) == 0:
		return (0, 0.0, 0.0)
	depths = [float(d[0][3]) / 256.0 for d in defects if float(d[0][3]) / 256.0 >= DEFECT_DEPTH_THRESHOLD * 0.25]
	if not depths:
		return (0, 0.0, 0.0)
	depths = np.array(depths, dtype=np.float32)
	return (int(len(depths)), float(np.mean(depths)), float(np.max(depths)))


TEMPLATE_BANK = {
	"T": [_T_TEMPLATE] + _rotations(_T_TEMPLATE),
	"Skew": [_SKew_TEMPLATE] + _rotations(_SKew_TEMPLATE),
	"L": [_L_TEMPLATE] + _rotations(_L_TEMPLATE),
}
TEMPLATE_BITMAPS = {}
TEMPLATE_DEFECT_COUNTS = {}
TEMPLATE_DEFECT_PROFILES = {}
for _label, _templates in TEMPLATE_BANK.items():
	_BITMAPS = []
	_DEFECT_COUNTS = []
	_PROFILES = []
	for _template in _templates:
		_BITMAPS.append(_contour_to_canvas(_template, size=TEMPLATE_CANVAS_SIZE))
		_DEFECT_COUNTS.append(_convex_defect_count(_template))
		_PROFILES.append(_convex_defect_profile(_template))
	TEMPLATE_BITMAPS[_label] = _BITMAPS
	TEMPLATE_DEFECT_COUNTS[_label] = int(round(np.median(_DEFECT_COUNTS))) if _DEFECT_COUNTS else 0
	TEMPLATE_DEFECT_PROFILES[_label] = _PROFILES


def update_feed():
	global DEBUG_TICK, SHOW_DEBUG_TEXT
	if not app_open:
		return

	if camera is None or not camera.isOpened():
		return

	ret, frame = camera.read()
	if not ret:
		_log_debug("Can't receive frame. Exiting...")
		stop_scanning()
		return

	display_frame = annotate_shape(frame)
	if SHOW_DEBUG_TEXT:
		for idx, msg in enumerate(DEBUG_INFO[:DEBUG_PRINT_LINES]):
			cv2.putText(
				display_frame,
				msg,
				(10, 18 + 18 * idx),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.65,
				(0, 255, 255),
				2,
				cv2.LINE_AA,
			)

	status = f"Debug: {'ON' if SHOW_DEBUG_TEXT else 'OFF'}"
	cv2.putText(
		display_frame,
		status,
		(10, frame.shape[0] - 18),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.55,
		(255, 255, 0) if SHOW_DEBUG_TEXT else (180, 180, 180),
		2,
		cv2.LINE_AA,
	)

	DEBUG_TICK += 1
	if DEBUG_TICK % DEBUG_PRINT_EVERY == 0:
		_log_debug(" | ".join(DEBUG_INFO))

	_set_tk_image(camera_label, display_frame)
	if DEBUG_MASK is not None:
		_set_tk_image(debug_label, DEBUG_MASK)
	else:
		_set_tk_image(debug_label, display_frame)

	if app_open:
		window.after(PROCESS_INTERVAL_MS, update_feed)


	


def annotate_shape(frame):
	working = frame.copy()
	frame_area = working.shape[0] * working.shape[1]
	global background_frame, background_calibrated, calibration_frames, DEBUG_INFO, DEBUG_MASK, FG_STABLE_MASK
	near_min_area = max(MIN_SHAPE_AREA, int(frame_area * MIN_SHAPE_AREA_RATIO_NEAR))
	far_min_area = max(MIN_SHAPE_AREA_FAR, int(frame_area * MIN_SHAPE_AREA_RATIO_FAR))
	DEBUG_INFO = [
		f"frame={frame.shape[1]}x{frame.shape[0]}",
		f"calibrated={background_calibrated}",
	]

	global_min_area = far_min_area

	if not background_calibrated:
		if background_frame is None:
			background_frame = np.float32(frame)
		else:
			background_frame += frame
		calibration_frames += 1

		if calibration_frames >= CALIBRATION_FRAME_COUNT:
			background_frame = np.clip(background_frame / float(calibration_frames), 0, 255).astype(np.uint8)
			background_calibrated = True

	if background_calibrated:
		DEBUG_INFO.append("bg_model: yes")
		frame_lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
		bg_lab = cv2.cvtColor(background_frame, cv2.COLOR_BGR2LAB)
		diff = cv2.absdiff(frame_lab, bg_lab)
		diff_mag = np.sqrt(np.sum(diff.astype(np.float32) ** 2, axis=2))
		_, bg_mask = cv2.threshold(diff_mag.astype(np.uint8), DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
	else:
		DEBUG_INFO.append("bg_model: no")
		bg_mask = np.zeros((working.shape[0], working.shape[1]), dtype=np.uint8)
	bg_kernel = np.ones((BG_OPEN_SIZE, BG_OPEN_SIZE), np.uint8)
	bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, bg_kernel)
	bg_close_kernel = np.ones((BG_CLOSE_SIZE, BG_CLOSE_SIZE), np.uint8)
	bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, bg_close_kernel)

	gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	fg_learning_rate = 0.01 if calibration_frames < FG_INIT_FRAMES else 0.0
	_, moving_raw = cv2.threshold(FG_SUBTRACTOR.apply(blurred, learningRate=fg_learning_rate), FG_THRESHOLD, 255, cv2.THRESH_BINARY)
	fg_kernel = np.ones((3, 3), np.uint8)
	moving_raw = cv2.morphologyEx(moving_raw, cv2.MORPH_OPEN, fg_kernel)
	moving_raw = cv2.morphologyEx(moving_raw, cv2.MORPH_CLOSE, fg_kernel)
	moving_count = int(np.count_nonzero(moving_raw))
	DEBUG_INFO.append(f"fg learning={fg_learning_rate}")
	DEBUG_INFO.append(f"moving raw={moving_count}")

	raw_float = moving_raw.astype(np.float32) / 255.0

	if FG_STABLE_MASK is None:
		FG_STABLE_MASK = raw_float.copy()
	else:
		FG_STABLE_MASK = FG_STABLE_MASK * FG_DECAY
		FG_STABLE_MASK = cv2.addWeighted(raw_float, FG_ALPHA, FG_STABLE_MASK, 1.0 - FG_ALPHA, 0.0)
	_, stable_move = cv2.threshold(FG_STABLE_MASK, 0.50, 1.0, cv2.THRESH_BINARY)
	moving_mask = (stable_move * 255).astype(np.uint8)
	DEBUG_INFO.append(f"moving stable={int(np.count_nonzero(moving_mask))}")
	DEBUG_INFO.append(f"bg_mask used={int(np.count_nonzero(bg_mask))}")
	edges = cv2.Canny(blurred, EDGE_LOW, EDGE_HIGH)
	kernel = np.ones((5, 5), np.uint8)
	edge_mask = cv2.dilate(edges, kernel, iterations=1)
	bg_gate = cv2.dilate(bg_mask, np.ones((BG_GATE_DILATE, BG_GATE_DILATE), np.uint8), iterations=1)
	shape_mask = cv2.bitwise_or(bg_mask, cv2.bitwise_and(edge_mask, bg_gate))
	DEBUG_INFO.append(f"bg_gate={int(np.count_nonzero(bg_gate))}")

	foreground_gate = moving_mask
	if background_calibrated:
		foreground_gate = cv2.bitwise_or(foreground_gate, bg_mask)
	shape_mask = cv2.bitwise_and(shape_mask, cv2.morphologyEx(foreground_gate, cv2.MORPH_DILATE, np.ones((7, 7), np.uint8)))
	DEBUG_INFO.append(f"foreground_gate={int(np.count_nonzero(foreground_gate))}")

	shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_OPEN, kernel)
	shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, kernel)
	shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
	shape_mask = cv2.morphologyEx(shape_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
	shape_mask = cv2.medianBlur(shape_mask, 5)
	_, shape_mask = cv2.threshold(shape_mask, 127, 255, cv2.THRESH_BINARY)
	mask_pixels = int(np.count_nonzero(shape_mask))
	mask_area_min = max(MIN_SHAPE_AREA, int(frame_area * 0.0002))
	DEBUG_INFO.append(f"mask pixels={mask_pixels}")
	DEBUG_INFO.append(f"target area range={int(global_min_area)}..{int(frame_area * MAX_SHAPE_AREA_RATIO)}")
	DEBUG_MASK = cv2.cvtColor(shape_mask, cv2.COLOR_GRAY2BGR)

	contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	DEBUG_INFO.append(f"contours mask={len(contours)}")

	if not contours:
		DEBUG_INFO.append("fallback: edges")
		contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		DEBUG_MASK = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
		DEBUG_INFO.append(f"contours edges={len(contours)}")

	if not contours:
		DEBUG_INFO.append("fallback: loose")
		_, loose = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(loose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		DEBUG_MASK = cv2.cvtColor(loose, cv2.COLOR_GRAY2BGR)
		DEBUG_INFO.append(f"contours loose={len(contours)}")

	if not contours:
		DEBUG_INFO.append("fallback: bg+motion")
		_, stable_bg_motion = cv2.threshold(cv2.bitwise_or(bg_mask, moving_raw), 127, 255, cv2.THRESH_BINARY)
		contours, _ = cv2.findContours(stable_bg_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		DEBUG_MASK = cv2.cvtColor(stable_bg_motion, cv2.COLOR_GRAY2BGR)
		DEBUG_INFO.append(f"contours bg+motion={len(contours)}")

	if not contours:
		DEBUG_INFO.append("no contours found")
		return working

	max_area = frame_area * MAX_SHAPE_AREA_RATIO
	min_area = global_min_area
	fallback = None
	relaxed_fallback = None
	candidates = []
	valid_count = 0
	relaxed_count = 0
	for c in contours:
		area = cv2.contourArea(c)
		area_ratio = area / float(frame_area) if frame_area else 0.0
		is_far = area_ratio <= FAR_AREA_RATIO_THRESHOLD
		per_contour_min_area = far_min_area if is_far else near_min_area
		per_contour_solidity = SOLIDITY_MIN_FAR if is_far else SOLIDITY_MIN
		per_contour_extent = EXTENT_MIN_FAR if is_far else EXTENT_MIN
		if area <= per_contour_min_area or area > max_area:
			if area > per_contour_min_area and area <= max_area:
				if fallback is None or area > fallback[0]:
					fallback = (area, c)
			continue

		relaxed_count += 1
		if relaxed_fallback is None or area > relaxed_fallback[0]:
			relaxed_fallback = (area, c)

		hull = cv2.convexHull(c)
		hull_area = cv2.contourArea(hull)
		if hull_area <= 0:
			continue
		solidity = area / hull_area
		x, y, w, h = cv2.boundingRect(c)
		if w <= 0 or h <= 0:
			continue
		extent = area / float(w * h)
		if solidity < per_contour_solidity or extent < per_contour_extent:
			continue

		valid_count += 1
		candidates.append((area, c))

	DEBUG_INFO.append(f"contours valid={valid_count}")
	DEBUG_INFO.append(f"contours relaxed={relaxed_count}")

	using_strict_shape = True
	if not candidates:
		if relaxed_fallback is not None:
			DEBUG_INFO.append("using relaxed contour")
			largest = relaxed_fallback[1]
			using_strict_shape = False
		elif fallback is not None:
			DEBUG_INFO.append("using fallback contour")
			largest = fallback[1]
			using_strict_shape = False
		else:
			DEBUG_INFO.append("using largest contour")
			largest = max(contours, key=cv2.contourArea)
			using_strict_shape = False
	else:
		DEBUG_INFO.append("using filtered contour")
		largest = max(candidates, key=lambda t: t[0])[1]
		using_strict_shape = True
		DEBUG_INFO.append(f"strict shape accepted area={int(cv2.contourArea(largest))}")

	area = cv2.contourArea(largest)
	DEBUG_INFO.append(f"chosen area={int(area)}")
	x, y, w, h = cv2.boundingRect(largest)
	x = max(0, x - SHAPE_AREA_PADDING)
	y = max(0, y - SHAPE_AREA_PADDING)
	w = min(working.shape[1] - x, w + SHAPE_AREA_PADDING * 2)
	h = min(working.shape[0] - y, h + SHAPE_AREA_PADDING * 2)
	cv2.rectangle(working, (x, y), (x + w, y + h), (0, 255, 0), 2)
	if using_strict_shape:
		shape_area_ratio = area / float(frame_area) if frame_area else 0.0
		score_threshold = SHAPE_SCORE_THRESHOLD_FAR if shape_area_ratio <= FAR_AREA_RATIO_THRESHOLD else SHAPE_SCORE_THRESHOLD
		shape_label = classify_shape(largest, score_threshold=score_threshold)
		DEBUG_INFO.append(f"score threshold={score_threshold:.2f}")
	else:
		shape_label = "Unknown"
	color_label = classify_color(working, largest)
	shape_label = f"{shape_label} / {color_label}"
	DEBUG_INFO.append(f"label={shape_label}")
	cv2.putText(
		working,
		shape_label,
		(x + 6, max(20, y - 10)),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.6,
		(0, 255, 0),
		2,
		cv2.LINE_AA,
	)
	return working


def classify_color(frame, contour):
	if contour is None or len(contour) < 3:
		return "Unknown"
	region = frame.copy()
	mask = np.zeros(frame.shape[:2], dtype=np.uint8)
	cv2.drawContours(mask, [contour], -1, 255, -1)
	masked = cv2.bitwise_and(region, region, mask=mask)
	hsv = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)
	v = hsv[:, :, 2][mask > 0]
	if v.size == 0:
		return "Unknown"
	if np.mean(v) <= COLOR_BLACK_V:
		return COLOR_BLACK

	s = hsv[:, :, 1][mask > 0]
	h = hsv[:, :, 0][mask > 0]
	valid = (s >= COLOR_LOW_SAT)
	if np.count_nonzero(valid) == 0:
		avg_v = np.mean(v)
		return COLOR_WHITE if avg_v > 190 else COLOR_GRAY
	hue = float(np.median(h[valid]))
	sat = float(np.mean(s[valid]))
	val = float(np.mean(v[valid]))

	if val < 50:
		return COLOR_BLACK
	if sat < 45 and val > 190:
		return COLOR_WHITE
	if sat < 55:
		return COLOR_GRAY
	if hue < 10 or hue >= 170:
		return "Red"
	if hue < 25:
		return "Orange"
	if hue < 35:
		return "Yellow"
	if hue < 78:
		return "Green"
	if hue < 95:
		return "Cyan"
	if hue < 115:
		return "Blue"
	if hue < 150:
		return "Purple"
	return "Magenta"


def classify_shape(contour, score_threshold=SHAPE_SCORE_THRESHOLD):
	best_label = "Unknown"
	best_score = 1e9
	second_score = 1e9
	best_label_raw = 1e9
	second_label_raw = 1e9
	second_label = "Unknown"
	normalized = contour
	if normalized is None or len(normalized) < 3:
		return best_label
	epsilon = 0.030 * cv2.arcLength(normalized, True)
	normalized = cv2.approxPolyDP(normalized, epsilon, True)
	candidate_bitmap = _contour_to_canvas(normalized, size=TEMPLATE_CANVAS_SIZE)
	if candidate_bitmap is None:
		candidate_bitmap = np.zeros((TEMPLATE_CANVAS_SIZE, TEMPLATE_CANVAS_SIZE), dtype=np.uint8)
	candidate_profile = _convex_defect_profile(normalized)

	candidate_defect_count = _convex_defect_count(normalized)
	for label, templates in TEMPLATE_BANK.items():
		label_best_score = 1e9
		template_bitmaps = TEMPLATE_BITMAPS.get(label, [])
		template_profiles = TEMPLATE_DEFECT_PROFILES.get(label, [])
		for idx, template in enumerate(templates):
			match_score = cv2.matchShapes(normalized, template, cv2.CONTOURS_MATCH_I1, 0.0)
			tb = template_bitmaps[idx] if idx < len(template_bitmaps) else None
			if tb is not None:
				diff = cv2.countNonZero(cv2.absdiff(candidate_bitmap, tb))
				raster_score = diff / float(TEMPLATE_CANVAS_SIZE * TEMPLATE_CANVAS_SIZE)
				match_score = min(match_score, raster_score)

			profile_penalty = 0.0
			if idx < len(template_profiles) and label in ("L", "Skew"):
				t_count, t_mean, t_max = template_profiles[idx]
				c_count, c_mean, c_max = candidate_profile
				profile_penalty += abs(c_count - t_count) * DEFECT_PENALTY
				profile_penalty += abs(c_mean - t_mean) * DEFECT_MEAN_PENALTY
				profile_penalty += abs(c_max - t_max) * DEFECT_MEAN_PENALTY
			profile_penalty = min(profile_penalty, PROFILE_PENALTY_MAX)
			score = match_score + profile_penalty
			label_best_score = min(label_best_score, score)
		label_name = label
		if label_best_score < 1e9:
			geometry_delta = abs(candidate_defect_count - TEMPLATE_DEFECT_COUNTS.get(label_name, 0)) * DEFECT_PENALTY
			combined_score = label_best_score + geometry_delta
			if combined_score < second_score:
				if combined_score < best_score:
					second_score = best_score
					second_label = best_label
					second_label_raw = best_label_raw
					best_score = combined_score
					best_label_raw = label_best_score
					best_label = label_name
				elif combined_score < second_score:
					second_score = combined_score
					second_label = label_name
					second_label_raw = label_best_score
			DEBUG_INFO.append(f"classify:{label_name} best={label_best_score:.3f} geom={geometry_delta:.3f} combined={combined_score:.3f}")

	if best_score > score_threshold:
		return "Unknown"

		# L vs Skew tie-break uses concavity profile from hull defects.
	if second_label in ("L", "Skew") and best_label in ("L", "Skew") and (second_score - best_score) < SHAPE_SCORE_GAP:
		best_profiles = TEMPLATE_DEFECT_PROFILES.get(best_label, [])
		second_profiles = TEMPLATE_DEFECT_PROFILES.get(second_label, [])
		best_profile = min(best_profiles, key=lambda p: (abs(candidate_defect_count - p[0]) + abs(candidate_profile[0] - p[0]))) if best_profiles else (0, 0.0, 0.0)
		second_profile = min(second_profiles, key=lambda p: (abs(candidate_defect_count - p[0]) + abs(candidate_profile[0] - p[0]))) if second_profiles else (0, 0.0, 0.0)
		best_defect_delta = abs(candidate_profile[0] - best_profile[0]) + abs(candidate_profile[1] - best_profile[1]) + abs(candidate_profile[2] - best_profile[2])
		second_defect_delta = abs(candidate_profile[0] - second_profile[0]) + abs(candidate_profile[1] - second_profile[1]) + abs(candidate_profile[2] - second_profile[2])
		if second_defect_delta + L_SKEW_PROFILE_SWAP_MARGIN < best_defect_delta:
			best_label = second_label
			best_score, second_score = second_score, best_score
			best_label_raw, second_label_raw = second_label_raw, best_label_raw
			DEBUG_INFO.append(f"classify tie-break: {best_label} closer in defect count")

	if second_label_raw < 1e9:
		DEBUG_INFO.append(f"classify raw scores: best={best_label_raw:.3f} second={second_label_raw:.3f} gap={second_label_raw - best_label_raw:.3f}")
	if second_score - best_score < SHAPE_SCORE_GAP:
		DEBUG_INFO.append(f"classify gap={second_score - best_score:.4f}")

	return f"{best_label} ({best_score:.3f})"


def stop_scanning():
	global camera, running, app_open, DEBUG_LOG_FILE
	if not running:
		return

	running = False
	if camera is not None:
		camera.release()
		camera = None

	if DEBUG_LOG_FILE is not None:
		try:
			DEBUG_LOG_FILE.close()
		except Exception:
			pass
		DEBUG_LOG_FILE = None


def stop_app():
	global app_open, running
	app_open = False
	running = False
	stop_scanning()
	window.destroy()


def toggle_debug_overlay():
	global SHOW_DEBUG_TEXT
	SHOW_DEBUG_TEXT = not SHOW_DEBUG_TEXT
	if 'debug_btn' in globals():
		debug_btn.config(text="Hide debug overlay" if SHOW_DEBUG_TEXT else "Show debug overlay")


window = tk.Tk()
window.title("PartSorter Webcam Scanner")

view_frame = tk.Frame(window)
view_frame.pack(padx=12, pady=8)
camera_label = tk.Label(view_frame, text="Camera Feed", compound=tk.TOP)
camera_label.grid(row=0, column=0, padx=6)
debug_label = tk.Label(view_frame, text="Detection Debug", compound=tk.TOP)
debug_label.grid(row=0, column=1, padx=6)

controls = tk.Frame(window)
controls.pack(padx=12, pady=8, fill=tk.X)
debug_btn = tk.Button(controls, text="Show debug overlay", command=toggle_debug_overlay)
debug_btn.pack(side=tk.LEFT, padx=8)

window.protocol("WM_DELETE_WINDOW", stop_app)

if camera.isOpened():
	camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
	running = True
	window.after(PROCESS_INTERVAL_MS, update_feed)
else:
	running = False
	_log_debug("Cannot open camera")

window.mainloop()


'''
def bumper(sensor):
	def bumpy():
		while not sensor.get_sample():
			pass
		return True
	return bumpy


def prep():
	global brick
	global motor_shoulder
	global motor_elbow
	global touch_shoulder
	global touch_elbow
	try:
		brick = nxt.locator.find()
	except nxt.locator.BrickNotFoundError:
		if sys.flags.interactive:
			return
		else:
			sys.exit(0)
	motor_shoulder = brick.get_motor(nxt.motor.Port.A)
	motor_elbow = brick.get_motor(nxt.motor.Port.B)
	touch_shoulder = brick.get_sensor(nxt.sensor.Port.S1, nxt.sensor.generic.Touch)
	touch_elbow = brick.get_sensor(nxt.sensor.Port.S2, nxt.sensor.generic.Touch)

def cleanup():
	motor_shoulder.idle()
	motor_elbow.idle()
	brick.close()

def home():
	motor_elbow.turn(15, 360, stop_turn = bumper(touch_elbow))
	motor_shoulder.turn(-15, 360, stop_turn = bumper(touch_shoulder))

def push_left():
	motor_shoulder.turn(30, 150) 
	motor_elbow.turn(-30, 150)
def push_right():
	motor_elbow.turn(-30, 250)
	motor_shoulder.turn(30, 100)
	motor_elbow.turn(5, 90)
	motor_shoulder.turn(-30, 90) 
def push_off():
	motor_elbow.turn(-30, 250)
	motor_shoulder.turn(30, 110) 
	motor_elbow.turn(30, 40)
	motor_shoulder.turn(-70, 50) 
prep()
home()
time.sleep(1)
"""
push_right()
time.sleep(1)
home()
time.sleep(1)
push_left()
time.sleep(1)
home()
time.sleep(1)
push_off()
time.sleep(1)


home()
time.sleep(1)
"""
cleanup() # for when you're done.
'''
