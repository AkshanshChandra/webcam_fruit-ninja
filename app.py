"""App: shared game context and the per-frame pipeline
(camera frame -> mirror -> tracking -> vignette -> state update/draw).
It does no window I/O itself, which keeps it testable headlessly."""

import time

import cv2
import numpy as np

import config
from fruit import build_fruit_types, make_bomb
from game_states import MODE_SURVIVAL, MenuState
from hand_tracker import Blade
from render import Vignette, draw_text, text_sprite


def canvas_size(frame_w, frame_h):
    h = config.CANVAS_HEIGHT
    return int(round(frame_w * h / frame_h)) // 2 * 2, h


def build_logo(scale):
    """Big gradient title with a katana slash through it."""
    size = int(118 * scale)
    a = text_sprite(config.GAME_TITLE, size, style="black", gradient=((60, 240, 255), (20, 110, 255)),
                    outline=(30, 10, 40), outline_w=int(9 * scale))
    h, w = a.shape[:2]
    pad = int(30 * scale)
    logo = np.zeros((h + pad * 2, w + pad * 2, 4), np.uint8)
    logo[pad:pad + h, pad:pad + w] = a
    # slash: glow + bright core, kept inside the alpha of the canvas
    p0 = (int(pad * 0.4), int(logo.shape[0] * 0.78))
    p1 = (logo.shape[1] - int(pad * 0.4), int(logo.shape[0] * 0.30))
    glow = np.zeros_like(logo)
    cv2.line(glow, p0, p1, (255, 220, 120, 200), max(4, int(16 * scale)), cv2.LINE_AA)
    glow = cv2.GaussianBlur(glow, (0, 0), 6 * scale)
    core = np.zeros_like(logo)
    cv2.line(core, p0, p1, (255, 255, 255, 255), max(2, int(4 * scale)), cv2.LINE_AA)
    for layer in (glow, core):
        la = layer[:, :, 3:4].astype(np.float32) / 255
        logo[:, :, :3] = (layer[:, :, :3] * la + logo[:, :, :3] * (1 - la)).astype(np.uint8)
        logo[:, :, 3] = np.maximum(logo[:, :, 3], layer[:, :, 3])
    return logo


class App:
    def __init__(self, w, h, tracker, audio, leaderboard):
        self.w, self.h = w, h
        self.scale = h / 720.0
        self.tracker = tracker
        self.audio = audio
        self.leaderboard = leaderboard
        self.vignette = Vignette(w, h)
        self.blade = Blade(self.scale)
        self.fruit_types = build_fruit_types(self.scale)
        self.bomb_radius = int(config.FRUIT_RADIUS * self.scale * 0.95)
        self.bomb_sprite = make_bomb(self.bomb_radius)
        self.logo = build_logo(self.scale)
        icon_src = next(t for t in self.fruit_types if t.name == "apple").sprite
        isz = int(54 * self.scale)
        self.life_icon = cv2.resize(icon_src, (isz, isz), interpolation=cv2.INTER_AREA)
        lost = self.life_icon.copy()
        gray = cv2.cvtColor(lost[:, :, :3], cv2.COLOR_BGR2GRAY)
        lost[:, :, :3] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) // 2
        lost[:, :, 3] = lost[:, :, 3] // 2
        self.life_icon_lost = lost

        self.mode = MODE_SURVIVAL
        self.quit = False
        self.toggle_fullscreen = False
        self.show_debug = False
        self.time = time.monotonic()
        self.last_hand_time = -1e9
        self.hand_present = False
        self.mouse = None          # (x, y, button_down) in canvas coords
        self.mouse_blade = False
        self._swoosh_cool = 0.0
        self._fps = 0.0
        self.state = None
        self.change(MenuState(self))

    def change(self, state):
        self.state = state
        state.enter()

    # --- input -------------------------------------------------------------
    def on_key(self, key):
        if key is None:
            return
        if not self.state.text_input:
            if key == "m":
                self.audio.toggle_mute()
                return
            if key == "f":
                self.toggle_fullscreen = True
                return
            if key == "d":
                self.show_debug = not self.show_debug
                return
        self.state.on_key(key)

    def on_mouse(self, x, y, down):
        self.mouse = (x, y, down)

    # --- frame pipeline ----------------------------------------------------
    def prepare(self, cam_frame):
        """Resize to the canvas and mirror. Returns a new array we can draw on."""
        fh, fw = cam_frame.shape[:2]
        if (fw, fh) != (self.w, self.h):
            cam_frame = cv2.resize(cam_frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return cv2.flip(cam_frame, 1)

    def step(self, cam_frame, dt, now, key=None, new_frame=True):
        """Advance one frame and return the rendered canvas."""
        dt = min(max(dt, 0.0), 0.05)
        self.time = now
        if dt > 0:
            self._fps = self._fps * 0.92 + (1.0 / dt) * 0.08
        canvas = self.prepare(cam_frame)
        if new_frame:
            self.tracker.submit(canvas)
        self.vignette.apply(canvas)

        self.on_key(key)
        self._update_blade(now, dt)
        self.state.update(dt, now)
        self.state.draw(canvas, now)
        self._draw_overlay(canvas)
        return canvas

    def _update_blade(self, now, dt):
        sample = self.tracker.latest(self.w, self.h)
        self.hand_present = sample is not None
        if self.hand_present:
            self.last_hand_time = now
        self.mouse_blade = False
        if sample is None and self.mouse and self.mouse[2]:
            sample = (self.mouse[0], self.mouse[1], now)
            self.mouse_blade = True
            self.last_hand_time = now
        override = getattr(self.state, "blade_override", None)
        if sample is None and override:
            demo = override(now, dt)
            if demo is not None:
                sample = None if demo == "none" else demo
        self.blade.update(sample, now)

        self._swoosh_cool -= dt
        if self.blade.active and self.blade.speed / self.scale > 2200 and self._swoosh_cool <= 0:
            self.audio.play("swoosh", 0.7)
            self._swoosh_cool = 0.35

    def draw_blade(self, frame, now):
        self.blade.draw(frame, now)

    def _draw_overlay(self, frame):
        s = self.scale
        if self.audio.muted:
            draw_text(frame, "SOUND OFF (M)", (self.w // 2, int(20 * s)), int(18 * s), (160, 160, 160))
        if self.show_debug:
            draw_text(frame, f"{self._fps:4.1f} FPS", (self.w // 2, int(44 * s)), int(20 * s),
                      (120, 255, 120))
