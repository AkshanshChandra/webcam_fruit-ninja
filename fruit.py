"""Procedurally drawn fruit/bomb sprites, the flying objects, sliced halves
and the wave spawner."""

import math
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import config
from render import blit_center, rotate_sprite

SS = 3  # supersampling factor for smooth sprite edges


def _rgb(bgr, a=255):
    return (bgr[2], bgr[1], bgr[0], a)


def _finish(img, size):
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img)
    return np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])


def _shaded_ball(d, box, base, dark, light):
    """Radial-ish shading: dark rim, base body, soft highlight."""
    x0, y0, x1, y1 = box
    d.ellipse(box, fill=_rgb(dark))
    inset = (x1 - x0) * 0.05
    d.ellipse((x0 + inset * 0.6, y0 + inset * 0.2, x1 - inset * 1.4, y1 - inset * 1.6), fill=_rgb(base))
    w = x1 - x0
    d.ellipse((x0 + w * 0.2, y0 + w * 0.14, x0 + w * 0.48, y0 + w * 0.36), fill=_rgb(light, 150))


def _highlight(img, box, strength=170):
    """Glossy specular blob."""
    hl = Image.new("RGBA", img.size, (0, 0, 0, 0))
    x0, y0, x1, y1 = box
    w = x1 - x0
    ImageDraw.Draw(hl).ellipse((x0 + w * 0.22, y0 + w * 0.16, x0 + w * 0.42, y0 + w * 0.3),
                               fill=(255, 255, 255, strength))
    hl = hl.filter(ImageFilter.GaussianBlur(w * 0.03))
    return Image.alpha_composite(img, hl)


def make_apple(r):
    size = int(r * 2.5)
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    R = r * SS
    body = (c - R, c - R * 0.92, c + R, c + R * 1.02)
    _shaded_ball(d, body, (40, 30, 215), (30, 15, 120), (120, 120, 255))
    d.ellipse((c - R * 0.25, c - R * 0.95, c + R * 0.25, c - R * 0.7), fill=_rgb((25, 15, 110)))
    d.line((c, c - R * 0.8, c + R * 0.12, c - R * 1.2), fill=_rgb((30, 60, 90)), width=int(R * 0.12))
    d.ellipse((c + R * 0.1, c - R * 1.25, c + R * 0.6, c - R * 0.98), fill=_rgb((60, 170, 60)))
    return _finish(_highlight(img, body), size)


def make_orange(r):
    size = int(r * 2.4)
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    R = r * SS
    body = (c - R, c - R, c + R, c + R)
    _shaded_ball(d, body, (20, 140, 255), (10, 80, 200), (120, 210, 255))
    rng = random.Random(7)
    for _ in range(70):
        a = rng.random() * math.tau
        rr = math.sqrt(rng.random()) * R * 0.85
        x, y = c + math.cos(a) * rr, c + math.sin(a) * rr
        d.ellipse((x - R * 0.025, y - R * 0.025, x + R * 0.025, y + R * 0.025), fill=_rgb((10, 105, 225), 160))
    d.ellipse((c - R * 0.1, c - R * 0.98, c + R * 0.1, c - R * 0.82), fill=_rgb((30, 90, 60)))
    d.ellipse((c + R * 0.02, c - R * 1.2, c + R * 0.45, c - R * 0.9), fill=_rgb((50, 160, 50)))
    return _finish(_highlight(img, body), size)


def make_watermelon(r):
    size = int(r * 2.3)
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    R = r * SS
    body = (c - R, c - R * 0.86, c + R, c + R * 0.86)
    d.ellipse(body, fill=_rgb((40, 110, 30)))
    stripes = Image.new("L", (S, S), 0)
    sd = ImageDraw.Draw(stripes)
    for k in range(-3, 4):
        x = c + k * R * 0.3
        pts = [(x + math.sin(t * 0.9) * R * 0.06, c - R + t * R * 0.1) for t in range(21)]
        sd.line(pts, fill=255, width=int(R * 0.11))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse(body, fill=255)
    stripes = Image.fromarray(np.minimum(np.array(stripes), np.array(mask)))
    img.paste(Image.new("RGBA", (S, S), _rgb((20, 60, 15))), (0, 0), stripes)
    img = _highlight(img, (c - R, c - R, c + R, c + R), 110)
    return _finish(img, size)


def make_banana(r):
    size = int(r * 2.6)
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    R = r * SS
    outer = (c - R * 1.15, c - R * 1.55, c + R * 1.15, c + R * 0.75)
    inner = (c - R * 0.95, c - R * 1.9, c + R * 0.95, c + R * 0.08)
    shape = Image.new("L", (S, S), 0)
    sd = ImageDraw.Draw(shape)
    sd.chord(outer, 15, 165, fill=255)
    sd.chord(inner, 10, 170, fill=0)
    shade = Image.new("L", (S, S), 0)
    ImageDraw.Draw(shade).chord((outer[0], outer[1] + R * 0.18, outer[2], outer[3]), 25, 155, fill=255)
    shade = Image.fromarray(np.minimum(np.array(shade), np.array(shape)))
    img.paste(Image.new("RGBA", (S, S), _rgb((20, 170, 220))), (0, 0), shape)
    img.paste(Image.new("RGBA", (S, S), _rgb((40, 220, 255))), (0, 0), shade.filter(ImageFilter.GaussianBlur(R * 0.04)))
    # brown tips
    for ang in (17, 163):
        a = math.radians(ang)
        tx = c + math.cos(a) * R * 1.07
        ty = c - R * 0.4 + math.sin(a) * R * 1.07
        d.ellipse((tx - R * 0.09, ty - R * 0.09, tx + R * 0.09, ty + R * 0.09), fill=_rgb((30, 60, 90)))
    img = img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, R * 0.38), Image.BICUBIC)
    return _finish(img, size)


def make_cross_section(r, rind, flesh, seed=None, ring=None, seeds=0):
    """Cut face of a round fruit: rind ring, flesh, optional seeds/segments."""
    size = int(r * 2.4)
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    R = r * SS
    d.ellipse((c - R, c - R, c + R, c + R), fill=_rgb(rind))
    if ring:
        d.ellipse((c - R * 0.9, c - R * 0.9, c + R * 0.9, c + R * 0.9), fill=_rgb(ring))
    d.ellipse((c - R * 0.82, c - R * 0.82, c + R * 0.82, c + R * 0.82), fill=_rgb(flesh))
    rng = random.Random(3)
    if seed and seeds:
        for i in range(seeds):
            a = i / seeds * math.tau + rng.random() * 0.3
            rr = R * (0.35 + rng.random() * 0.3)
            x, y = c + math.cos(a) * rr, c + math.sin(a) * rr
            d.ellipse((x - R * 0.05, y - R * 0.08, x + R * 0.05, y + R * 0.08), fill=_rgb(seed))
    return _finish(img, size)


def make_orange_section(r):
    size = int(r * 2.4)
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    R = r * SS
    d.ellipse((c - R, c - R, c + R, c + R), fill=_rgb((10, 120, 245)))
    d.ellipse((c - R * 0.9, c - R * 0.9, c + R * 0.9, c + R * 0.9), fill=_rgb((190, 235, 255)))
    for i in range(10):
        a0, a1 = i * 36 + 3, i * 36 + 33
        d.pieslice((c - R * 0.84, c - R * 0.84, c + R * 0.84, c + R * 0.84), a0, a1, fill=_rgb((40, 170, 255)))
    d.ellipse((c - R * 0.1, c - R * 0.1, c + R * 0.1, c + R * 0.1), fill=_rgb((190, 235, 255)))
    return _finish(img, size)


def make_bomb(r):
    size = int(r * 2.6)
    S = size * SS
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    c = S / 2
    R = r * SS
    # red danger glow
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((c - R * 1.12, c - R * 1.0, c + R * 1.12, c + R * 1.24), fill=(255, 30, 30, 120))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(R * 0.1)))
    d = ImageDraw.Draw(img)
    body = (c - R * 0.92, c - R * 0.8, c + R * 0.92, c + R * 1.04)
    d.ellipse(body, fill=(15, 15, 20, 255))
    d.ellipse((body[0] + R * 0.06, body[1] + R * 0.04, body[2] - R * 0.18, body[3] - R * 0.2), fill=(45, 45, 55, 255))
    d.rectangle((c - R * 0.22, c - R * 0.98, c + R * 0.22, c - R * 0.7), fill=(90, 90, 100, 255))
    d.line([(c, c - R * 0.98), (c + R * 0.15, c - R * 1.15), (c + R * 0.38, c - R * 1.22)],
           fill=(170, 130, 80, 255), width=int(R * 0.09), joint="curve")
    # skull-ish X mark
    w = R * 0.22
    d.line((c - w, c + R * 0.05 - w, c + w, c + R * 0.05 + w), fill=(230, 40, 40, 255), width=int(R * 0.1))
    d.line((c - w, c + R * 0.05 + w, c + w, c + R * 0.05 - w), fill=(230, 40, 40, 255), width=int(R * 0.1))
    img = _highlight(img, body, 120)
    return _finish(img, size)


# Fuse tip offset from sprite center (in units of radius), used for the spark.
BOMB_FUSE_TIP = (0.38, -1.22)


class FruitType:
    def __init__(self, name, radius, sprite, section, juice, flesh, weight):
        self.name = name
        self.radius = radius
        self.sprite = sprite
        self.section = section     # sprite used for the cut halves
        self.juice = juice         # BGR juice color
        self.flesh = flesh         # BGR flesh color (cut edge on halves)
        self.weight = weight


def build_fruit_types(scale):
    r = int(config.FRUIT_RADIUS * scale)
    rw = int(r * 1.25)
    return [
        FruitType("apple", r, make_apple(r),
                  make_cross_section(r, (40, 30, 210), (190, 235, 250), seed=(40, 70, 110), seeds=5),
                  (120, 210, 245), (190, 235, 250), 3),
        FruitType("orange", r, make_orange(r), make_orange_section(r),
                  (20, 160, 255), (40, 170, 255), 3),
        FruitType("watermelon", rw, make_watermelon(rw),
                  make_cross_section(rw, (40, 110, 30), (80, 60, 245), seed=(20, 20, 30), ring=(200, 240, 220), seeds=11),
                  (80, 50, 240), (80, 60, 245), 2),
        FruitType("banana", r, make_banana(r), None, (120, 235, 255), (170, 240, 255), 2),
    ]


class Flyer:
    """Shared ballistic motion + rotation."""

    def __init__(self, x, y, vx, vy, radius, gravity):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.gravity = gravity
        self.angle = random.uniform(0, 360)
        self.spin = random.uniform(-160, 160)
        self.age = 0.0

    def step(self, dt):
        self.age += dt
        self.vy += self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.spin * dt


class Fruit(Flyer):
    is_bomb = False

    def __init__(self, ftype, x, y, vx, vy, gravity):
        super().__init__(x, y, vx, vy, ftype.radius, gravity)
        self.type = ftype
        self.sliced = False

    def draw(self, frame, now):
        blit_center(frame, rotate_sprite(self.type.sprite, self.angle), self.x, self.y)


class Bomb(Flyer):
    is_bomb = True

    def __init__(self, sprite, radius, x, y, vx, vy, gravity):
        super().__init__(x, y, vx, vy, radius, gravity)
        self.sprite = sprite
        self.spin *= 0.4
        self.sliced = False

    def fuse_tip(self):
        fx, fy = BOMB_FUSE_TIP
        a = math.radians(-self.angle)
        ox, oy = fx * self.radius, fy * self.radius
        return (self.x + ox * math.cos(a) - oy * math.sin(a),
                self.y + ox * math.sin(a) + oy * math.cos(a))

    def draw(self, frame, now):
        blit_center(frame, rotate_sprite(self.sprite, self.angle), self.x, self.y)
        tx, ty = self.fuse_tip()
        flick = 0.6 + 0.4 * math.sin(now * 40 + self.x)
        r = max(2, int(self.radius * 0.22 * flick))
        p = (int(tx), int(ty))
        cv2.circle(frame, p, int(r * 1.8), (0, 120, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, p, r, (120, 240, 255), -1, cv2.LINE_AA)
        for k in range(4):
            a = now * 25 + k * math.pi / 2
            cv2.line(frame, p, (int(tx + math.cos(a) * r * 3), int(ty + math.sin(a) * r * 3)),
                     (200, 250, 255), 1, cv2.LINE_AA)


class FruitHalf(Flyer):
    def __init__(self, sprite, x, y, vx, vy, radius, gravity, angle, spin):
        super().__init__(x, y, vx, vy, radius, gravity)
        self.sprite = sprite
        self.angle = angle
        self.spin = spin

    def draw(self, frame, now):
        blit_center(frame, rotate_sprite(self.sprite, self.angle), self.x, self.y)


def split_sprites(fruit, cut_angle_deg):
    """Cut a fruit sprite along a line through its center. Returns the two half
    sprites (in the sprite's own unrotated frame) and the cut normal in world
    coordinates, pointing towards half A."""
    ft = fruit.type
    base = ft.section if ft.section is not None else ft.sprite
    h, w = base.shape[:2]
    # Cut direction relative to the sprite (sprite is drawn rotated by fruit.angle CCW).
    local = math.radians(cut_angle_deg + fruit.angle)
    nx, ny = -math.sin(local), math.cos(local)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    side = (xs - w / 2) * nx + (ys - h / 2) * ny
    halves = []
    for sign in (1, -1):
        s = base.copy()
        keep = (side * sign) > 0
        s[:, :, 3] = np.where(keep, s[:, :, 3], 0)
        if ft.section is None:
            # banana: paint a flesh-colored band along the cut edge
            band = (np.abs(side) < max(3, ft.radius * 0.12)) & keep & (s[:, :, 3] > 0)
            s[band, :3] = ft.flesh
        halves.append(s)
    wa = math.radians(cut_angle_deg)
    return halves, (-math.sin(wa), math.cos(wa))


def make_halves(fruit, cut_angle_deg, gravity, scale):
    (a, b), (nx, ny) = split_sprites(fruit, cut_angle_deg)
    push = 160 * scale
    out = []
    for sprite, sgn in ((a, 1), (b, -1)):
        out.append(FruitHalf(
            sprite,
            fruit.x + nx * sgn * fruit.radius * 0.1, fruit.y + ny * sgn * fruit.radius * 0.1,
            fruit.vx * 0.6 + nx * sgn * push, min(fruit.vy, 0) * 0.4 + ny * sgn * push - 80 * scale,
            fruit.radius, gravity, fruit.angle, fruit.spin + sgn * random.uniform(120, 260)))
    return out


def lerp(a, b, t):
    return a + (b - a) * max(0.0, min(1.0, t))


class Spawner:
    """Launches waves of fruit (and the occasional bomb) from below the screen.
    Difficulty ramps with `elapsed` (seconds into the round)."""

    def __init__(self, w, h, scale, fruit_types, bomb_sprite, bomb_radius,
                 ramp_seconds=config.RAMP_SECONDS, bombs=True):
        self.w, self.h, self.scale = w, h, scale
        self.types = fruit_types
        self.weights = [t.weight for t in fruit_types]
        self.bomb_sprite = bomb_sprite
        self.bomb_radius = bomb_radius
        self.gravity = config.GRAVITY * scale
        self.ramp_seconds = ramp_seconds
        self.bombs = bombs
        self.timer = 0.8
        self._queue = []   # (delay, factory) for staggered launches inside a wave

    def difficulty(self, elapsed):
        return min(1.0, elapsed / self.ramp_seconds)

    def update(self, dt, elapsed):
        out = []
        for item in self._queue:
            item[0] -= dt
        while self._queue and self._queue[0][0] <= 0:
            out.append(self._queue.pop(0)[1]())
        self.timer -= dt
        if self.timer <= 0:
            d = self.difficulty(elapsed)
            interval = lerp(*config.SPAWN_INTERVAL, d)
            self.timer = interval * random.uniform(0.8, 1.2)
            self._queue_wave(d)
        return out

    def _queue_wave(self, d):
        max_n = int(round(lerp(*config.WAVE_SIZE, d)))
        n = random.randint(1, max(1, max_n))
        bomb = self.bombs and random.random() < lerp(*config.BOMB_CHANCE, d)
        kinds = ["fruit"] * n + (["bomb"] if bomb else [])
        random.shuffle(kinds)
        stagger = random.choice([0.0, 0.12, 0.2])
        for i, k in enumerate(kinds):
            self._queue.append([i * stagger, self.make_bomb if k == "bomb" else self.make_fruit])
        self._queue.sort(key=lambda q: q[0])

    def _launch(self, radius):
        w, h = self.w, self.h
        x = random.uniform(w * 0.15, w * 0.85)
        apex = random.uniform(h * 0.12, h * 0.42)
        vy = -math.sqrt(2 * self.gravity * (h + radius - apex))
        t_air = 2 * -vy / self.gravity
        target_x = random.uniform(w * 0.25, w * 0.75)
        vx = (target_x - x) / t_air * random.uniform(0.8, 1.6)
        return x, h + radius, vx, vy

    def make_fruit(self):
        ft = random.choices(self.types, self.weights)[0]
        x, y, vx, vy = self._launch(ft.radius)
        return Fruit(ft, x, y, vx, vy, self.gravity)

    def make_bomb(self):
        x, y, vx, vy = self._launch(self.bomb_radius)
        return Bomb(self.bomb_sprite, self.bomb_radius, x, y, vx, vy, self.gravity)


def segment_hits_circle(p0, p1, cx, cy, r):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / L2))
    px, py = x0 + dx * t, y0 + dy * t
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r
