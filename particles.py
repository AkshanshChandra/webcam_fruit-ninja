"""Visual effects: juice droplets, juice splats, bomb explosion, animated
popup text and screen flash/shake."""

import math
import random

import cv2
import numpy as np

from render import blit_center, draw_text


def _shade(color, f):
    return tuple(int(max(0, min(255, c * f))) for c in color)


class Droplet:
    __slots__ = ("x", "y", "vx", "vy", "r", "life", "max_life", "color", "g")

    def __init__(self, x, y, vx, vy, r, life, color, g):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.r, self.life, self.max_life, self.color, self.g = r, life, life, color, g

    def step(self, dt):
        self.life -= dt
        self.vy += self.g * dt
        self.vx *= 0.985
        self.x += self.vx * dt
        self.y += self.vy * dt
        return self.life > 0

    def draw(self, frame):
        k = self.life / self.max_life
        r = int(self.r * (0.35 + 0.65 * k))
        if r >= 1:
            p = (int(self.x), int(self.y))
            cv2.circle(frame, p, r, self.color, -1, cv2.LINE_AA)
            if r >= 3:
                cv2.circle(frame, (p[0] - r // 3, p[1] - r // 3), max(1, r // 3),
                           _shade(self.color, 1.4), -1, cv2.LINE_AA)


class Ember(Droplet):
    """Additive glowing spark (used for explosions and fuse sparks)."""

    def draw(self, frame):
        k = self.life / self.max_life
        r = max(1, int(self.r * k))
        cv2.circle(frame, (int(self.x), int(self.y)), r, _shade(self.color, 0.5 + 0.5 * k), -1, cv2.LINE_AA)


def _splat_sprite(radius, color, seed):
    rng = random.Random(seed)
    size = int(radius * 3.2)
    img = np.zeros((size, size, 4), np.uint8)
    c = size // 2
    col = tuple(int(v) for v in color)
    cv2.circle(img, (c, c), int(radius * 0.75), col + (255,), -1, cv2.LINE_AA)
    for _ in range(14):
        a = rng.random() * math.tau
        d = radius * rng.uniform(0.4, 1.2)
        rr = int(radius * rng.uniform(0.12, 0.35))
        cv2.circle(img, (int(c + math.cos(a) * d), int(c + math.sin(a) * d)), rr, col + (255,), -1, cv2.LINE_AA)
        if rng.random() < 0.5:
            d2 = d * 1.35
            cv2.circle(img, (int(c + math.cos(a) * d2), int(c + math.sin(a) * d2)), max(2, rr // 2),
                       col + (255,), -1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img[:, :, 3] = (img[:, :, 3].astype(np.float32) * 0.7).astype(np.uint8)
    return img


class Splat:
    """Juice stain left on the 'screen' that fades out."""

    def __init__(self, x, y, radius, color, life=1.6):
        self.x, self.y = x, y
        self.sprite = _splat_sprite(radius, color, random.randint(0, 10_000))
        self.life = self.max_life = life

    def step(self, dt):
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        blit_center(frame, self.sprite, self.x, self.y, min(1.0, self.life / (self.max_life * 0.6)))


class Shockwave:
    def __init__(self, x, y, max_r, color, life=0.6, thickness=10):
        self.x, self.y, self.max_r, self.color = x, y, max_r, color
        self.life = self.max_life = life
        self.thickness = thickness

    def step(self, dt):
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        k = 1 - self.life / self.max_life
        r = int(self.max_r * (1 - (1 - k) ** 3))
        t = max(1, int(self.thickness * (1 - k)))
        cv2.circle(frame, (int(self.x), int(self.y)), r, _shade(self.color, 1 - k * 0.6), t, cv2.LINE_AA)


class Popup:
    """Animated text: pops in with an overshoot, floats up and fades out."""

    def __init__(self, text, x, y, size, color, life=1.0, rise=60, gradient=None, delay=0.0):
        self.text, self.x, self.y, self.size, self.color = text, x, y, size, color
        self.life = self.max_life = life
        self.rise = rise
        self.gradient = gradient
        self.delay = delay

    def step(self, dt):
        if self.delay > 0:
            self.delay -= dt
            return True
        self.life -= dt
        return self.life > 0

    def draw(self, frame):
        if self.delay > 0:
            return
        t = 1 - self.life / self.max_life
        if t < 0.18:
            s = 0.3 + 1.0 * (t / 0.18)            # grow past full size
        elif t < 0.3:
            s = 1.3 - 0.3 * ((t - 0.18) / 0.12)   # settle back
        else:
            s = 1.0
        alpha = 1.0 if t < 0.65 else max(0.0, 1 - (t - 0.65) / 0.35)
        y = self.y - self.rise * t
        h, w = frame.shape[:2]
        x = min(max(self.x, w * 0.18), w * 0.82)
        draw_text(frame, self.text, (int(x), int(y)), self.size, self.color, alpha=alpha,
                  scale=s, gradient=self.gradient)


class Effects:
    """Container for all transient effects, plus screen flash and shake."""

    def __init__(self, scale):
        self.scale = scale
        self.splats = []
        self.particles = []
        self.rings = []
        self.popups = []
        self.flash_color = (255, 255, 255)
        self.flash = 0.0
        self.flash_decay = 2.0
        self.shake = 0.0

    def clear(self):
        self.splats.clear()
        self.particles.clear()
        self.rings.clear()
        self.popups.clear()
        self.flash = 0.0
        self.shake = 0.0

    def juice(self, x, y, color, radius, direction=None):
        s = self.scale
        g = 900 * s
        n = 26
        for _ in range(n):
            a = random.uniform(0, math.tau)
            sp = random.uniform(80, 520) * s
            vx, vy = math.cos(a) * sp, math.sin(a) * sp - 120 * s
            if direction is not None:
                vx += direction[0] * 260 * s
                vy += direction[1] * 260 * s
            col = _shade(color, random.uniform(0.75, 1.15))
            self.particles.append(Droplet(x, y, vx, vy, random.uniform(3, 9) * s,
                                          random.uniform(0.4, 0.9), col, g))
        self.splats.append(Splat(x + random.uniform(-10, 10) * s, y, radius * 1.1, _shade(color, 1.0)))
        self.rings.append(Shockwave(x, y, radius * 1.8, _shade(color, 1.2), 0.35, int(6 * s)))

    def spark(self, x, y):
        s = self.scale
        a = random.uniform(0, math.tau)
        sp = random.uniform(40, 160) * s
        self.particles.append(Ember(x, y, math.cos(a) * sp, math.sin(a) * sp - 40 * s,
                                    random.uniform(2, 4) * s, random.uniform(0.15, 0.35),
                                    random.choice([(80, 220, 255), (40, 160, 255), (200, 250, 255)]), 300 * s))

    def explosion(self, x, y):
        s = self.scale
        for _ in range(140):
            a = random.uniform(0, math.tau)
            sp = random.uniform(150, 1100) * s
            col = random.choice([(60, 200, 255), (30, 120, 255), (40, 60, 240), (200, 250, 255), (20, 20, 20)])
            self.particles.append(Ember(x, y, math.cos(a) * sp, math.sin(a) * sp,
                                        random.uniform(4, 14) * s, random.uniform(0.5, 1.4), col, 500 * s))
        self.rings.append(Shockwave(x, y, 520 * s, (120, 230, 255), 0.7, int(26 * s)))
        self.rings.append(Shockwave(x, y, 330 * s, (60, 80, 255), 0.9, int(16 * s)))
        self.flash_now((60, 60, 255), 0.7, decay=1.8)
        self.shake = 1.0

    def flash_now(self, color, amount, decay=2.0):
        self.flash_color = color
        self.flash = amount
        self.flash_decay = decay

    def popup(self, *args, **kw):
        self.popups.append(Popup(*args, **kw))

    def step(self, dt):
        self.splats = [p for p in self.splats if p.step(dt)]
        self.particles = [p for p in self.particles if p.step(dt)]
        self.rings = [p for p in self.rings if p.step(dt)]
        self.popups = [p for p in self.popups if p.step(dt)]
        self.flash = max(0.0, self.flash - dt * self.flash_decay)
        self.shake = max(0.0, self.shake - dt * 1.4)

    def draw_under(self, frame):
        for p in self.splats:
            p.draw(frame)

    def draw_over(self, frame):
        for p in self.rings:
            p.draw(frame)
        for p in self.particles:
            p.draw(frame)

    def draw_popups(self, frame):
        for p in self.popups:
            p.draw(frame)

    def shake_offset(self):
        if self.shake <= 0:
            return 0, 0
        m = 22 * self.scale * self.shake
        return int(random.uniform(-m, m)), int(random.uniform(-m, m))
