"""Fruit Ninja webcam edition - entry point.

    python main.py                # fullscreen, default webcam
    python main.py --windowed     # large window instead of fullscreen
    python main.py --camera 1     # pick another webcam
"""

import os

# Quieten MediaPipe/TFLite native logging (must be set before mediapipe loads).
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import sys
import time
import traceback

import cv2
import numpy as np

import config

# Arrow / special key codes differ per OS and HighGUI backend (cv2.waitKeyEx).
_SPECIAL_KEYS = {
    "up": {63232, 2490368, 65362, 16777235},
    "down": {63233, 2621440, 65364, 16777237},
    "left": {63234, 2424832, 65361, 16777234},
    "right": {63235, 2555904, 65363, 16777236},
    "enter": {13, 10, 65293, 16777220, 16777221},
    "backspace": {8, 127, 65288, 16777219},
    "esc": {27, 16777216},
    "space": {32},
}
MIN_FRAME_TIME = 1 / 60  # cap the render loop; the camera itself delivers ~30 fps

_CODE_TO_NAME = {code: name for name, codes in _SPECIAL_KEYS.items() for code in codes}


def decode_key(code):
    if code is None or code < 0:
        return None
    if code in _CODE_TO_NAME:
        return _CODE_TO_NAME[code]
    low = code & 0xFF
    if code < 256 and 33 <= low <= 126:
        return chr(low).lower()
    return None


def log_crash(tb):
    try:
        with open(config.CRASH_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n{tb}")
    except OSError:
        pass
    print(tb, file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description="Webcam Fruit Ninja")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--windowed", action="store_true", help="start in a large window instead of fullscreen")
    p.add_argument("--mute", action="store_true", help="start with sound muted")
    p.add_argument("--no-audio", action="store_true", help="disable the audio device entirely")
    p.add_argument("--debug", action="store_true", help="show FPS counter")
    return p.parse_args()


class Window:
    def __init__(self, fullscreen):
        self.fullscreen = fullscreen
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        self.apply()

    def apply(self):
        if self.fullscreen:
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(config.WINDOW_NAME, 1280, 720)

    def toggle(self):
        self.fullscreen = not self.fullscreen
        self.apply()

    def closed(self):
        try:
            return cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True


def error_screen(message, detail=""):
    from render import Vignette, draw_panel, draw_text
    w, h = 1280, 720
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (40, 20, 30)
    Vignette(w, h, 1.0, 0.5).apply(img)
    draw_panel(img, 140, 150, 1000, 420, fill=(30, 10, 25), fill_alpha=230, border=config.NEON_RED,
               border_w=4, radius=28)
    draw_text(img, "CAMERA PROBLEM", (w // 2, 220), 64, config.NEON_RED, style="black")
    y = 310
    for line in _wrap(message, 60):
        draw_text(img, line, (w // 2, y), 28, config.WHITE)
        y += 40
    for line in _wrap(detail, 70):
        draw_text(img, line, (w // 2, y + 10), 22, (190, 190, 190))
        y += 32
    draw_text(img, "R  retry        ESC  quit", (w // 2, 520), 34, config.NEON_YELLOW, style="black")
    return img


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if len(cur) + len(wd) + 1 > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines


def open_camera_with_retry(camera, window):
    """Open the webcam; on failure show an on-screen error until fixed or quit."""
    from camera import CameraError
    while True:
        try:
            return camera.open()
        except CameraError as exc:
            msg = str(exc)
            print(f"[camera] {msg}", file=sys.stderr)
            parts = msg.split(". ", 1)
            img = error_screen(parts[0] + ".", parts[1] if len(parts) > 1 else "")
            while True:
                cv2.imshow(config.WINDOW_NAME, img)
                k = decode_key(cv2.waitKeyEx(50))
                if k == "r":
                    break
                if k in ("esc", "q") or window.closed():
                    return None


def run(args):
    from app import App, canvas_size
    from game_states import MenuState
    from audio import Audio
    from camera import Camera
    from hand_tracker import HandTracker
    from leaderboard import Leaderboard

    window = Window(fullscreen=not args.windowed)
    from render import draw_text
    loading = np.zeros((720, 1280, 3), np.uint8)
    draw_text(loading, "LOADING...", (640, 360), 60, config.NEON_CYAN, style="black")
    cv2.imshow(config.WINDOW_NAME, loading)
    cv2.waitKey(1)

    camera = Camera(args.camera)
    size = open_camera_with_retry(camera, window)
    if size is None:
        return 1
    w, h = canvas_size(*size)

    leaderboard = Leaderboard()
    audio = Audio(enabled=not args.no_audio)
    if args.mute:
        audio.toggle_mute()
    tracker = HandTracker()
    app = App(w, h, tracker, audio, leaderboard)
    app.show_debug = args.debug

    def on_mouse(event, x, y, flags, _):
        app.on_mouse(x, y, bool(flags & cv2.EVENT_FLAG_LBUTTON))
    cv2.setMouseCallback(config.WINDOW_NAME, on_mouse)

    last = time.monotonic()
    last_frame_id = -1
    errors = []
    try:
        while not app.quit:
            frame_id, frame = camera.read()
            if camera.stalled():
                print("[camera] stream stalled, reconnecting...", file=sys.stderr)
                camera.close()
                if open_camera_with_retry(camera, window) is None:
                    break
                continue
            now = time.monotonic()
            dt = now - last
            last = now
            key = decode_key(cv2.waitKeyEx(1))
            try:
                canvas = app.step(frame, dt, now, key, new_frame=frame_id != last_frame_id)
            except Exception:
                # Keep the event running: log the bug and fall back to the menu.
                errors = [t for t in errors if now - t < 10] + [now]
                if len(errors) > 5:
                    raise
                log_crash(traceback.format_exc())
                app.blade.reset()
                app.change(MenuState(app))
                continue
            last_frame_id = frame_id
            cv2.imshow(config.WINDOW_NAME, canvas)
            if app.toggle_fullscreen:
                app.toggle_fullscreen = False
                window.toggle()
            if window.closed():
                break
            spare = MIN_FRAME_TIME - (time.monotonic() - now)
            if spare > 0.002:
                time.sleep(spare)
    finally:
        tracker.close()
        audio.close()
        camera.close()
        cv2.destroyAllWindows()
    return 0


def main():
    args = parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        return 0
    except Exception:
        log_crash(traceback.format_exc())
        print(f"Fruit Ninja crashed - details written to {config.CRASH_LOG_FILE}. "
              f"Leaderboard data is safe in {config.LEADERBOARD_FILE}.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
