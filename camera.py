"""Threaded webcam capture so a slow read never stalls rendering, with
clear errors for missing cameras / denied permissions and auto-reconnect."""

import platform
import threading
import time

import cv2

import config


class CameraError(RuntimeError):
    pass


def _help_text():
    if platform.system() == "Darwin":
        return ("Grant camera access: System Settings > Privacy & Security > Camera, "
                "enable your terminal/IDE app, then restart it.")
    if platform.system() == "Windows":
        return "Check Settings > Privacy > Camera and close other apps using the webcam."
    return "Check that the webcam is connected and not used by another app (/dev/video*)."


class Camera:
    def __init__(self, index=0):
        self.index = index
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._last_frame_time = 0.0
        self._running = False
        self._thread = None

    def _open_capture(self):
        backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else (
            cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Could not open webcam #{self.index}. {_help_text()}")
        w, h = config.CAMERA_REQUEST_SIZE
        if platform.system() != "Darwin":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def open(self, timeout=4.0):
        """Open the device and wait for the first frame (raises CameraError)."""
        self.close()
        self._cap = self._open_capture()
        deadline = time.monotonic() + timeout
        ok, frame = False, None
        while time.monotonic() < deadline:
            ok, frame = self._cap.read()
            if ok and frame is not None and frame.size:
                break
            time.sleep(0.05)
        if not ok or frame is None:
            self._cap.release()
            self._cap = None
            raise CameraError(f"Webcam #{self.index} opened but sent no frames. {_help_text()}")
        with self._lock:
            self._frame = frame
            self._frame_id += 1
            self._last_frame_time = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return frame.shape[1], frame.shape[0]

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._last_frame_time = time.monotonic()

    def read(self):
        """(frame_id, frame) of the newest frame; frame may be shared, copy before drawing."""
        with self._lock:
            return self._frame_id, self._frame

    def stalled(self, limit=2.5):
        return self._running and time.monotonic() - self._last_frame_time > limit

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
