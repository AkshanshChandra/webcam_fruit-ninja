"""Synthesised sound effects mixed in real time through sounddevice.
If no audio device is available the game simply runs silently."""

import threading

import numpy as np

SR = 44100


def _t(dur):
    return np.arange(int(SR * dur), dtype=np.float32) / SR


def _env(n, attack=0.005, release=None, curve=4.0):
    e = np.ones(n, np.float32)
    a = max(1, int(SR * attack))
    e[:a] = np.linspace(0, 1, a, dtype=np.float32)
    tail = np.linspace(1, 0, n - a, dtype=np.float32) ** curve
    e[a:] = tail
    return e


def _sweep(f0, f1, dur, wave="sine"):
    t = _t(dur)
    f = np.geomspace(f0, f1, len(t)).astype(np.float32)
    ph = 2 * np.pi * np.cumsum(f) / SR
    if wave == "square":
        return np.sign(np.sin(ph)).astype(np.float32) * 0.5
    if wave == "tri":
        return (2 / np.pi * np.arcsin(np.sin(ph))).astype(np.float32)
    return np.sin(ph).astype(np.float32)


def _noise(dur, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, int(SR * dur)).astype(np.float32)


def _lowpass(x, k):
    """Cheap one-pole low-pass, k in (0,1]: smaller = darker."""
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += k * (v - acc)
        y[i] = acc
    return y


def _highpass(x):
    return np.diff(x, prepend=x[:1])


def _norm(x, peak=0.9):
    m = np.max(np.abs(x)) or 1.0
    return (x / m * peak).astype(np.float32)


def build_sounds():
    s = {}
    # slice: bright noisy "shhk" + a quick descending squelch
    n = _noise(0.22, 1)
    swish = _highpass(n) * _env(len(n), 0.002, curve=3)
    squelch = _sweep(900, 180, 0.22, "tri") * _env(len(n), 0.004, curve=2) * 0.5
    s["slice"] = _norm(swish + squelch, 0.75)
    # swoosh of a fast blade swing
    n = _noise(0.25, 2)
    lp = _lowpass(n, 0.18)
    e = np.sin(np.linspace(0, np.pi, len(n), dtype=np.float32)) ** 2
    s["swoosh"] = _norm(lp * e, 0.35)
    # launch thump
    s["launch"] = _norm(_sweep(220, 70, 0.14) * _env(int(SR * 0.14), 0.002, curve=2), 0.35)
    # bomb explosion: rumbling low-passed noise + sub drop
    n = _noise(1.4, 3)
    boom = _lowpass(n, 0.04) * _env(len(n), 0.003, curve=2.5) * 3.0
    sub = _sweep(120, 30, 1.4) * _env(len(n), 0.003, curve=2)
    crack = _highpass(_noise(1.4, 4)) * _env(len(n), 0.001, curve=30)
    s["explosion"] = _norm(boom + sub + crack * 0.6, 0.95)
    # miss: sad descending bonk
    s["miss"] = _norm(_sweep(330, 160, 0.3, "square") * _env(int(SR * 0.3), 0.003, curve=2), 0.3)
    # countdown beeps
    s["beep"] = _norm(np.sin(2 * np.pi * 660 * _t(0.15)) * _env(int(SR * 0.15), 0.003, curve=1.5), 0.45)
    s["go"] = _norm((np.sin(2 * np.pi * 990 * _t(0.4)) + 0.4 * np.sin(2 * np.pi * 1980 * _t(0.4)))
                    * _env(int(SR * 0.4), 0.003, curve=1.5), 0.5)
    # combo: quick rising arpeggio
    notes = [523.25, 659.25, 783.99, 1046.5]
    parts = [np.sin(2 * np.pi * f * _t(0.09)) * _env(int(SR * 0.09), 0.002, curve=1.2) for f in notes]
    s["combo"] = _norm(np.concatenate(parts), 0.5)
    # menu tick / select
    s["tick"] = _norm(_sweep(1500, 1200, 0.04) * _env(int(SR * 0.04), 0.001), 0.3)
    s["select"] = _norm(np.concatenate([np.sin(2 * np.pi * f * _t(0.07)) * _env(int(SR * 0.07), 0.002, curve=1.2)
                                        for f in (660, 990)]), 0.45)
    # game over: descending triad
    notes = [523.25, 415.3, 329.63, 261.63]
    parts = [_sweep(f, f * 0.98, 0.16, "tri") * _env(int(SR * 0.16), 0.003, curve=1.3) for f in notes]
    s["gameover"] = _norm(np.concatenate(parts), 0.5)
    # new high score fanfare
    notes = [523.25, 659.25, 783.99, 1046.5, 783.99, 1046.5]
    parts = [np.sin(2 * np.pi * f * _t(0.11)) * _env(int(SR * 0.11), 0.003, curve=1.0) for f in notes]
    s["highscore"] = _norm(np.concatenate(parts), 0.5)
    return s


class Audio:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.muted = False
        self.error = None
        self._voices = []
        self._lock = threading.Lock()
        self._stream = None
        self.sounds = {}
        if not enabled:
            return
        try:
            self.sounds = build_sounds()
            import sounddevice as sd
            self._stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                                           blocksize=512, latency="low", callback=self._callback)
            self._stream.start()
        except Exception as exc:
            self.error = f"Audio disabled: {exc}"
            print(f"[audio] {self.error}")
            self._stream = None

    def _callback(self, outdata, frames, time_info, status):
        mix = np.zeros(frames, np.float32)
        with self._lock:
            alive = []
            for v in self._voices:
                data, pos, vol = v
                chunk = data[pos:pos + frames]
                mix[:len(chunk)] += chunk * vol
                v[1] = pos + frames
                if v[1] < len(data):
                    alive.append(v)
            self._voices = alive
        np.clip(mix, -1, 1, out=mix)
        outdata[:, 0] = mix

    def play(self, name, volume=1.0):
        if self._stream is None or self.muted:
            return
        data = self.sounds.get(name)
        if data is None:
            return
        with self._lock:
            if len(self._voices) < 16:
                self._voices.append([data, 0, volume])

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            with self._lock:
                self._voices.clear()

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
