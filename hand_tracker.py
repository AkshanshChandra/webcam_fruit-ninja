"""MediaPipe hand tracking (Tasks API, async live-stream mode) plus the Blade,
which turns fingertip samples into a glowing trail and slice segments."""

import math
import os
import threading
import time
import urllib.request
from collections import deque

import cv2
import numpy as np

import config

INDEX_FINGER_TIP = 8


def ensure_model(path=config.HAND_MODEL_FILE, url=config.HAND_MODEL_URL):
    """Return the model path, downloading it once if it is missing."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[hand] downloading hand model to {path} ...")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    return path


class HandTracker:
    """Runs MediaPipe HandLandmarker asynchronously. submit() never blocks the
    render loop; latest() returns the most recent fingertip in canvas pixels."""

    def __init__(self):
        self.error = None
        self._lock = threading.Lock()
        self._result = None          # (x_norm, y_norm, timestamp_s) or None
        self._result_time = 0.0
        self._last_ts = 0
        self._landmarker = None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            self._mp = mp
            options = vision.HandLandmarkerOptions(
                # CPU delegate: on macOS the default graph can try to use Metal and
                # abort the whole process when no GPU service is available.
                base_options=BaseOptions(model_asset_path=ensure_model(),
                                         delegate=BaseOptions.Delegate.CPU),
                running_mode=vision.RunningMode.LIVE_STREAM,
                num_hands=1,
                min_hand_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=config.HAND_PRESENCE_CONFIDENCE,
                min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
                result_callback=self._on_result,
            )
            self._landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as exc:  # missing model / no internet / broken install
            self.error = f"Hand tracking unavailable: {exc}"
            print(f"[hand] {self.error}")

    @property
    def available(self):
        return self._landmarker is not None

    def _on_result(self, result, image, timestamp_ms):
        point = None
        if result.hand_landmarks:
            tip = result.hand_landmarks[0][INDEX_FINGER_TIP]
            point = (tip.x, tip.y, timestamp_ms / 1000.0)
        with self._lock:
            self._result = point
            self._result_time = time.monotonic()

    def submit(self, frame_bgr):
        """Queue a (mirrored) BGR frame for detection."""
        if not self._landmarker:
            return
        h, w = frame_bgr.shape[:2]
        tw = config.TRACKER_INPUT_WIDTH
        small = cv2.resize(frame_bgr, (tw, int(h * tw / w)), interpolation=cv2.INTER_AREA) if w > tw else frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            self._landmarker.detect_async(image, ts)
        except Exception as exc:
            print(f"[hand] detect failed: {exc}")

    def latest(self, w, h):
        """Fingertip as (x_px, y_px, t) or None if no hand is currently seen."""
        with self._lock:
            r, rt = self._result, self._result_time
        if r is None or time.monotonic() - rt > 0.4:
            return None
        return (r[0] * w, r[1] * h, r[2])

    def close(self):
        if self._landmarker:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


class Blade:
    """Fingertip trail. update() is fed one sample per frame (or None) and
    exposes `segments`: the swipe segments created this frame with their speed."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.points = deque(maxlen=config.TRAIL_LENGTH)  # (x, y, t)
        self.segments = []                               # [(p0, p1, speed_px_s)]
        self.pos = None
        self.speed = 0.0
        self._last_sample_t = None

    def reset(self):
        self.points.clear()
        self.segments = []
        self.pos = None
        self.speed = 0.0
        self._last_sample_t = None

    @property
    def active(self):
        return self.pos is not None

    def update(self, sample, now):
        """sample: (x, y, t) or None. now: current monotonic time."""
        self.segments = []
        if sample is None:
            self.pos = None
            self.speed *= 0.5
        else:
            x, y, t = sample
            if self._last_sample_t is not None and t <= self._last_sample_t:
                pass  # no new tracker result this frame
            else:
                fresh = (self.pos is None or self._last_sample_t is None
                         or t - self._last_sample_t > config.MAX_SEGMENT_GAP)
                if fresh:
                    sx, sy = x, y
                    self.points.clear()
                else:
                    a = config.BLADE_SMOOTHING
                    sx = self.pos[0] + (x - self.pos[0]) * a
                    sy = self.pos[1] + (y - self.pos[1]) * a
                    dt = max(t - self._last_sample_t, 1e-3)
                    dist = math.hypot(sx - self.pos[0], sy - self.pos[1])
                    self.speed = dist / dt
                    self.segments.append((self.pos, (sx, sy), self.speed / self.scale))
                self.pos = (sx, sy)
                self._last_sample_t = t
                self.points.append((sx, sy, now))
        while self.points and now - self.points[0][2] > config.TRAIL_MAX_AGE:
            self.points.popleft()

    def slicing_segments(self):
        """Segments fast enough to cut (speed normalised to a 720p canvas)."""
        return [(p0, p1) for p0, p1, v in self.segments if v >= config.MIN_SLICE_SPEED]

    def draw(self, frame, now, glow_color=config.TRAIL_GLOW, core_color=config.TRAIL_CORE):
        pts = list(self.points)
        if len(pts) < 2 and self.pos is None:
            return
        s = self.scale
        fh, fw = frame.shape[:2]
        xs = [p[0] for p in pts] + ([self.pos[0]] if self.pos else [])
        ys = [p[1] for p in pts] + ([self.pos[1]] if self.pos else [])
        pad = int(40 * s)
        x1, y1 = max(int(min(xs)) - pad, 0), max(int(min(ys)) - pad, 0)
        x2, y2 = min(int(max(xs)) + pad, fw), min(int(max(ys)) + pad, fh)
        if x1 >= x2 or y1 >= y2:
            return
        roi = frame[y1:y2, x1:x2]
        glow = np.zeros_like(roi)
        core = np.zeros_like(roi)
        n = len(pts)
        for i in range(n - 1):
            (ax, ay, at), (bx, by, bt) = pts[i], pts[i + 1]
            age_fade = max(0.0, 1.0 - (now - bt) / config.TRAIL_MAX_AGE)
            pos_fade = (i + 1) / (n - 1)
            f = age_fade * (0.25 + 0.75 * pos_fade)
            if f <= 0.02:
                continue
            p0 = (int(ax - x1), int(ay - y1))
            p1 = (int(bx - x1), int(by - y1))
            gc = tuple(int(c * f) for c in glow_color)
            cc = tuple(int(c * f) for c in core_color)
            cv2.line(glow, p0, p1, gc, max(2, int((10 + 16 * pos_fade) * s)), cv2.LINE_AA)
            cv2.line(core, p0, p1, cc, max(1, int((2 + 7 * pos_fade) * s)), cv2.LINE_AA)
        if self.pos is not None:
            c = (int(self.pos[0] - x1), int(self.pos[1] - y1))
            cv2.circle(glow, c, int(16 * s), glow_color, -1, cv2.LINE_AA)
            cv2.circle(core, c, int(6 * s), core_color, -1, cv2.LINE_AA)
        k = max(3, int(21 * s) | 1)
        glow = cv2.GaussianBlur(glow, (k, k), 0)
        cv2.add(roi, glow, dst=roi)
        cv2.add(roi, glow // 2, dst=roi)
        cv2.add(roi, core, dst=roi)
