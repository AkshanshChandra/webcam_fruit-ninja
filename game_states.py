"""Screen state machine: Menu -> Countdown -> Play -> GameOver -> Leaderboard."""

import math
import random
import time

import cv2
import numpy as np

import config
from fruit import Spawner, make_halves, segment_hits_circle
from particles import Effects
from render import (blit, blit_center, darken, draw_panel, draw_text, rotate_sprite,
                    text_sprite, tint)

MODE_SURVIVAL = "survival"
MODE_TIMER = "timer"
MODE_LABELS = {MODE_SURVIVAL: "SURVIVAL", MODE_TIMER: "60s TIME ATTACK"}

COMBO_NAMES = {2: "DOUBLE SLICE!", 3: "TRIPLE SLICE!", 4: "QUAD SLICE!", 5: "PENTA SLICE!"}


def combo_bonus(n):
    """Total bonus for an n-fruit combo: 2->10, 3->30, 4->60, 5->100 ..."""
    return config.POINTS_PER_FRUIT * n * (n - 1) // 2


def combo_name(n):
    return COMBO_NAMES.get(n, f"MEGA COMBO x{n}!")


class World:
    """Flying fruit, halves and effects plus slice detection. Used by the menu
    (attract mode) and the actual game."""

    def __init__(self, app, bombs=True, ramp_seconds=config.RAMP_SECONDS):
        self.app = app
        s = app.scale
        self.spawner = Spawner(app.w, app.h, s, app.fruit_types, app.bomb_sprite, app.bomb_radius,
                               ramp_seconds=ramp_seconds, bombs=bombs)
        self.objects = []
        self.halves = []
        self.fx = Effects(s)

    def update(self, dt, elapsed, segments, spawn=True, allow_slice=True):
        events = []
        if spawn:
            for obj in self.spawner.update(dt, elapsed):
                self.objects.append(obj)
                self.app.audio.play("launch", 0.6)
        for o in self.objects:
            o.step(dt)
            if o.is_bomb and random.random() < 0.6:
                self.fx.spark(*o.fuse_tip())
        for hv in self.halves:
            hv.step(dt)
        self.halves = [hv for hv in self.halves if hv.y - hv.radius * 1.5 < self.app.h]

        if allow_slice and segments:
            for p0, p1 in segments:
                for o in self.objects:
                    if o.sliced:
                        continue
                    if segment_hits_circle(p0, p1, o.x, o.y, o.radius * config.HITBOX_SCALE):
                        o.sliced = True
                        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
                        if o.is_bomb:
                            events.append(("bomb", o))
                        else:
                            self._slice_fruit(o, math.degrees(math.atan2(dy, dx)), (dx, dy))
                            events.append(("slice", o))

        h = self.app.h
        keep = []
        for o in self.objects:
            if o.sliced:
                continue
            if o.vy > 0 and o.y - o.radius > h:
                if not o.is_bomb:
                    events.append(("miss", o))
                continue
            keep.append(o)
        self.objects = keep
        self.fx.step(dt)
        return events

    def _slice_fruit(self, fruit, angle, direction):
        self.halves.extend(make_halves(fruit, angle, fruit.gravity, self.app.scale))
        n = math.hypot(*direction) or 1.0
        self.fx.juice(fruit.x, fruit.y, fruit.type.juice, fruit.radius, (direction[0] / n, direction[1] / n))
        self.app.audio.play("slice")

    def draw(self, frame, now):
        self.fx.draw_under(frame)
        for hv in self.halves:
            hv.draw(frame, now)
        for o in self.objects:
            o.draw(frame, now)
        self.fx.draw_over(frame)


# ---------------------------------------------------------------------------

class State:
    text_input = False   # when True, letter keys are not treated as global shortcuts

    def __init__(self, app):
        self.app = app
        self.t = 0.0

    def enter(self):
        pass

    def on_key(self, key):
        pass

    def update(self, dt, now):
        self.t += dt

    def draw(self, frame, now):
        pass


def blink(t, period=1.0):
    return (t % period) < period * 0.65


def draw_hand_status(app, frame):
    s = app.scale
    x, y = int(24 * s), app.h - int(28 * s)
    if app.hand_present:
        cv2.circle(frame, (x, y), int(9 * s), config.NEON_GREEN, -1, cv2.LINE_AA)
        draw_text(frame, "HAND TRACKED", (x + int(18 * s), y), int(20 * s), config.NEON_GREEN, anchor="left")
    elif app.mouse_blade:
        cv2.circle(frame, (x, y), int(9 * s), config.NEON_YELLOW, -1, cv2.LINE_AA)
        draw_text(frame, "MOUSE BLADE", (x + int(18 * s), y), int(20 * s), config.NEON_YELLOW, anchor="left")
    else:
        on = blink(app.time, 1.2)
        cv2.circle(frame, (x, y), int(9 * s), (90, 90, 90) if on else (50, 50, 50), -1, cv2.LINE_AA)
        msg = "SHOW YOUR HAND TO THE CAMERA" if app.tracker.available else "HAND TRACKING OFF - HOLD MOUSE BUTTON"
        draw_text(frame, msg, (x + int(18 * s), y), int(20 * s), (200, 200, 200), anchor="left")


class DemoBlade:
    """Attract-mode 'ghost' blade that swipes through fruit when nobody plays."""

    def __init__(self, app):
        self.app = app
        self.swipe = None     # (start_time, duration, target, offset)
        self.cooldown = 0.5
        self.pos = (app.w * 0.5, app.h * 0.6)

    def sample(self, world, now, dt):
        s = self.app.scale
        if self.swipe is None:
            self.cooldown -= dt
            if self.cooldown <= 0:
                cands = [o for o in world.objects if not o.sliced and not o.is_bomb
                         and o.y < self.app.h * 0.75 and abs(o.vy) < 500 * s]
                if cands:
                    target = random.choice(cands)
                    a = random.uniform(0, math.tau)
                    d = 170 * s
                    self.swipe = (now, 0.16, target, (math.cos(a) * d, math.sin(a) * d))
            # idle drift toward screen centre
            cx, cy = self.app.w * 0.5, self.app.h * 0.55
            self.pos = (self.pos[0] + (cx - self.pos[0]) * dt, self.pos[1] + (cy - self.pos[1]) * dt)
            return None
        t0, dur, target, (ox, oy) = self.swipe
        k = (now - t0) / dur
        if k >= 1.0:
            self.swipe = None
            self.cooldown = random.uniform(0.25, 0.9)
            return None
        x = target.x - ox + 2 * ox * k
        y = target.y - oy + 2 * oy * k
        self.pos = (x, y)
        return (x, y, now)


# ---------------------------------------------------------------------------

class MenuState(State):
    OPTIONS = [MODE_SURVIVAL, MODE_TIMER, "leaderboard"]

    def __init__(self, app):
        super().__init__(app)
        self.world = World(app, bombs=False, ramp_seconds=1e9)
        self.world.spawner.timer = 0.3
        self.demo = DemoBlade(app)
        self.selected = self.OPTIONS.index(app.mode) if app.mode in self.OPTIONS else 0
        self.idle = 0.0
        self.quit_armed = 0.0

    def enter(self):
        self.app.blade.reset()

    def on_key(self, key):
        self.idle = 0.0
        if key in ("up", "w"):
            self.selected = (self.selected - 1) % len(self.OPTIONS)
            self.app.audio.play("tick")
        elif key in ("down", "s"):
            self.selected = (self.selected + 1) % len(self.OPTIONS)
            self.app.audio.play("tick")
        elif key in ("left", "right"):
            # quick toggle between the two modes
            self.selected = 1 if self.selected == 0 else 0
            self.app.audio.play("tick")
        elif key in ("space", "enter"):
            self.app.audio.play("select")
            choice = self.OPTIONS[self.selected]
            if choice == "leaderboard":
                self.app.change(LeaderboardState(self.app))
            else:
                self.app.mode = choice
                self.app.change(CountdownState(self.app))
        elif key == "l":
            self.app.change(LeaderboardState(self.app))
        elif key in ("esc", "q"):
            if self.quit_armed > 0:
                self.app.quit = True
            else:
                self.quit_armed = 2.0

    def blade_override(self, now, dt):
        """Menu uses a demo blade when no real hand/mouse is present."""
        if self.app.hand_present or self.app.mouse_blade:
            return None
        if now - self.app.last_hand_time < config.DEMO_BLADE_AFTER:
            return None
        return self.demo.sample(self.world, now, dt) or "none"

    def update(self, dt, now):
        super().update(dt, now)
        self.quit_armed = max(0.0, self.quit_armed - dt)
        if self.app.hand_present or self.app.mouse_blade:
            self.idle = 0.0
        else:
            self.idle += dt
        self.world.update(dt, 0, self.app.blade.slicing_segments())
        if self.idle > config.MENU_IDLE_TO_LEADERBOARD:
            self.app.change(LeaderboardState(self.app, attract=True))

    def draw(self, frame, now):
        app, s, w, h = self.app, self.app.scale, self.app.w, self.app.h
        self.world.draw(frame, now)
        app.draw_blade(frame, now)

        # Title
        bob = math.sin(self.t * 2.0) * 6 * s
        tilt = math.sin(self.t * 1.3) * 0.02
        title = app.logo
        title = rotate_sprite(title, math.degrees(tilt))
        blit_center(frame, title, w // 2, int(140 * s + bob))
        draw_text(frame, config.GAME_SUBTITLE, (w // 2, int(242 * s)), int(30 * s), config.NEON_CYAN,
                  style="black")

        best = app.leaderboard.top(1)
        if best:
            draw_text(frame, f"BEST  {best[0]['score']}  by  {best[0]['name']}", (w // 2, int(286 * s)),
                      int(24 * s), config.GOLD)

        # Options
        bw, bh = int(480 * s), int(62 * s)
        labels = {MODE_SURVIVAL: "SURVIVAL", MODE_TIMER: "60s TIME ATTACK", "leaderboard": "LEADERBOARD"}
        for i, opt in enumerate(self.OPTIONS):
            cy = int((350 + i * 78) * s)
            sel = i == self.selected
            pulse = 0.5 + 0.5 * math.sin(self.t * 6)
            border = config.NEON_PINK if sel else (90, 70, 90)
            draw_panel(frame, w // 2 - bw // 2, cy - bh // 2, bw, bh,
                       fill=(60, 20, 50) if sel else (30, 15, 25), fill_alpha=215 if sel else 160,
                       border=border, border_w=4 if sel else 2, radius=int(18 * s))
            col = config.WHITE if sel else (190, 180, 190)
            draw_text(frame, labels[opt], (w // 2, cy), int((34 if sel else 30) * s), col,
                      scale=1.0 + (0.04 * pulse if sel else 0))
            if sel:
                ax = w // 2 - bw // 2 - int(26 * s) - int(6 * s * pulse)
                pts = np.array([[ax, cy - int(14 * s)], [ax + int(20 * s), cy], [ax, cy + int(14 * s)]], np.int32)
                cv2.fillPoly(frame, [pts], config.NEON_PINK, cv2.LINE_AA)
                ax2 = w // 2 + bw // 2 + int(26 * s) + int(6 * s * pulse)
                pts2 = np.array([[ax2, cy - int(14 * s)], [ax2 - int(20 * s), cy], [ax2, cy + int(14 * s)]], np.int32)
                cv2.fillPoly(frame, [pts2], config.NEON_PINK, cv2.LINE_AA)

        desc = {
            MODE_SURVIVAL: "3 lives. Don't drop fruit. Never slice a bomb!",
            MODE_TIMER: "Slice as much as you can in 60 seconds. Bombs cost points.",
            "leaderboard": "See today's top ninjas.",
        }[self.OPTIONS[self.selected]]
        draw_text(frame, desc, (w // 2, int(588 * s)), int(24 * s), (230, 230, 230))
        if blink(self.t, 1.1):
            draw_text(frame, "PRESS SPACE TO START", (w // 2, int(636 * s)), int(36 * s), config.NEON_YELLOW,
                      style="black")
        draw_text(frame, "UP/DOWN choose   SPACE select   L leaderboard   M mute   F fullscreen",
                  (w - int(20 * s), h - int(28 * s)), int(17 * s), (170, 170, 170), anchor="right")
        if self.quit_armed > 0:
            draw_text(frame, "Press ESC again to quit", (w // 2, int(680 * s)), int(24 * s), config.NEON_RED)
        draw_hand_status(app, frame)


# ---------------------------------------------------------------------------

class CountdownState(State):
    STEPS = [("3", 0.8), ("2", 0.8), ("1", 0.8), ("GO!", 0.6)]

    def __init__(self, app):
        super().__init__(app)
        self.index = -1
        self.step_t = 0.0

    def enter(self):
        self.app.blade.reset()
        self._advance()

    def _advance(self):
        self.index += 1
        self.step_t = 0.0
        if self.index < len(self.STEPS):
            self.app.audio.play("go" if self.index == len(self.STEPS) - 1 else "beep")

    def on_key(self, key):
        if key == "esc":
            self.app.change(MenuState(self.app))

    def update(self, dt, now):
        super().update(dt, now)
        self.step_t += dt
        if self.step_t >= self.STEPS[self.index][1]:
            self._advance()
            if self.index >= len(self.STEPS):
                self.app.change(PlayState(self.app, self.app.mode))

    def draw(self, frame, now):
        app, s, w, h = self.app, self.app.scale, self.app.w, self.app.h
        app.draw_blade(frame, now)
        draw_text(frame, MODE_LABELS[app.mode], (w // 2, int(110 * s)), int(40 * s), config.NEON_CYAN,
                  style="black")
        draw_text(frame, "Get your hand in frame - swipe with your index finger!",
                  (w // 2, int(165 * s)), int(26 * s), config.WHITE)
        if self.index < len(self.STEPS):
            label, dur = self.STEPS[self.index]
            k = self.step_t / dur
            scale = 1.6 - 0.6 * min(1.0, k * 3)
            alpha = 1.0 if k < 0.7 else max(0.0, 1 - (k - 0.7) / 0.3)
            grad = ((80, 255, 255), (30, 140, 255)) if label != "GO!" else ((140, 255, 140), (40, 200, 60))
            draw_text(frame, label, (w // 2, h // 2 + int(30 * s)), int(200 * s), config.WHITE,
                      alpha=alpha, scale=scale, style="black", gradient=grad, outline_w=int(10 * s))
        draw_hand_status(app, frame)


# ---------------------------------------------------------------------------

class PlayState(State):
    def __init__(self, app, mode):
        super().__init__(app)
        self.mode = mode
        self.world = World(app, bombs=True)
        if mode == MODE_TIMER:
            self.world.spawner.ramp_seconds = config.TIMER_MODE_SECONDS
        self.score = 0
        self.lives = config.START_LIVES
        self.elapsed = 0.0
        self.time_left = float(config.TIMER_MODE_SECONDS)
        self.sliced = 0
        self.best_combo = 0
        self.combo_count = 0
        self.combo_last = -10.0
        self.combo_popup = None
        self.score_pop = 0.0
        self.paused = False
        self.ending = None      # (reason, time)
        self.end_timer = 0.0
        self.miss_marks = []    # (x, life)

    def enter(self):
        self.app.blade.reset()

    def on_key(self, key):
        if key == "p" and not self.ending:
            self.paused = not self.paused
        elif key == "esc":
            if self.paused:
                self.app.change(MenuState(self.app))
            else:
                self.paused = True

    # --- scoring -----------------------------------------------------------
    def _add_points(self, pts):
        self.score = max(0, self.score + pts)
        self.score_pop = 1.0

    def _on_slice(self, fruit, now):
        s = self.app.scale
        self.sliced += 1
        self._add_points(config.POINTS_PER_FRUIT)
        self.world.fx.popup(f"+{config.POINTS_PER_FRUIT}", fruit.x, fruit.y - fruit.radius, int(34 * s),
                            config.WHITE, life=0.7, rise=50 * s)
        if self.elapsed - self.combo_last <= config.COMBO_WINDOW:
            self.combo_count += 1
        else:
            self.combo_count = 1
        self.combo_last = self.elapsed
        n = self.combo_count
        if n >= 2:
            extra = combo_bonus(n) - combo_bonus(n - 1)
            self._add_points(extra)
            self.best_combo = max(self.best_combo, n)
            for old in self.combo_popup or ():
                if old in self.world.fx.popups:
                    self.world.fx.popups.remove(old)
            colors = [config.NEON_YELLOW, config.NEON_ORANGE, config.NEON_PINK, config.NEON_CYAN]
            self.world.fx.popup(combo_name(n), fruit.x, fruit.y - 60 * s, int((50 + 6 * min(n, 6)) * s),
                                colors[(n - 2) % len(colors)], life=1.2, rise=40 * s,
                                gradient=((120, 255, 255), (40, 120, 255)) if n >= 3 else None)
            self.world.fx.popup(f"+{combo_bonus(n)} BONUS", fruit.x, fruit.y - 5 * s, int(30 * s),
                                config.NEON_GREEN, life=1.0, rise=30 * s, delay=0.05)
            self.combo_popup = self.world.fx.popups[-2:]
            if n >= 3:
                self.app.audio.play("combo")

    def _on_bomb(self, bomb):
        s = self.app.scale
        fx = self.world.fx
        fx.explosion(bomb.x, bomb.y)
        self.app.audio.play("explosion")
        if self.mode == MODE_SURVIVAL or config.TIMER_BOMB_ENDS_ROUND:
            self._end("bomb")
        else:
            self._add_points(-config.TIMER_BOMB_PENALTY)
            fx.shake = 0.6
            fx.popup(f"-{config.TIMER_BOMB_PENALTY}", bomb.x, bomb.y, int(60 * s), config.NEON_RED,
                     life=1.2, rise=60 * s)

    def _on_miss(self, fruit):
        if self.mode != MODE_SURVIVAL:
            return
        self.lives -= 1
        self.app.audio.play("miss")
        self.miss_marks.append([min(max(fruit.x, 40), self.app.w - 40), 1.2])
        self.world.fx.flash_now((40, 40, 220), 0.25, decay=1.5)
        if self.lives <= 0:
            self._end("lives")

    def _end(self, reason):
        if self.ending:
            return
        self.ending = reason
        self.end_timer = 2.2 if reason == "bomb" else 1.6
        if reason != "bomb":
            self.app.audio.play("gameover")

    # --- loop ----------------------------------------------------------------
    def update(self, dt, now):
        if self.paused:
            return
        super().update(dt, now)
        self.score_pop = max(0.0, self.score_pop - dt * 4)
        for m in self.miss_marks:
            m[1] -= dt
        self.miss_marks = [m for m in self.miss_marks if m[1] > 0]

        if self.ending:
            # world keeps moving (slow-mo for bombs) but nothing can be scored
            slow = 0.35 if self.ending == "bomb" else 1.0
            self.world.update(dt * slow, self.elapsed, [], spawn=False, allow_slice=False)
            self.end_timer -= dt
            if self.end_timer <= 0:
                self.app.change(GameOverState(self.app, self.mode, self.score, self.ending,
                                              self.sliced, self.best_combo))
            return

        self.elapsed += dt
        if self.mode == MODE_TIMER:
            prev = self.time_left
            self.time_left = max(0.0, config.TIMER_MODE_SECONDS - self.elapsed)
            if math.ceil(prev) != math.ceil(self.time_left) and 0 < self.time_left <= 5:
                self.app.audio.play("beep", 0.6)
            if self.time_left <= 0:
                self._end("time")
                return

        events = self.world.update(dt, self.elapsed, self.app.blade.slicing_segments())
        for kind, obj in events:
            if self.ending and kind != "bomb":
                continue
            if kind == "slice":
                self._on_slice(obj, now)
            elif kind == "bomb":
                self._on_bomb(obj)
            elif kind == "miss":
                self._on_miss(obj)

    def draw(self, frame, now):
        app, s, w, h = self.app, self.app.scale, self.app.w, self.app.h
        self.world.draw(frame, now)
        if not self.ending:
            app.draw_blade(frame, now)
        self.world.fx.draw_popups(frame)

        for x, life in self.miss_marks:
            a = min(1.0, life / 0.4)
            draw_text(frame, "X", (int(x), h - int(50 * s)), int(70 * s), config.NEON_RED, alpha=a,
                      style="black")

        self._draw_hud(frame)

        fx = self.world.fx
        if fx.flash > 0:
            tint(frame, fx.flash_color, min(0.9, fx.flash))
        if self.ending:
            k = 1.0 - self.end_timer / (2.2 if self.ending == "bomb" else 1.6)
            text = {"bomb": "KABOOM!", "lives": "GAME OVER", "time": "TIME'S UP!"}[self.ending]
            sc = min(1.0, 0.4 + k * 3)
            draw_text(frame, text, (w // 2, h // 2), int(130 * s), config.WHITE, scale=sc, style="black",
                      gradient=((120, 255, 255), (30, 140, 255)),
                      outline_w=int(8 * s))
        if self.paused:
            darken(frame, 0.45)
            draw_text(frame, "PAUSED", (w // 2, h // 2 - int(30 * s)), int(110 * s), config.WHITE, style="black")
            draw_text(frame, "P to resume   ESC to quit to menu", (w // 2, h // 2 + int(60 * s)),
                      int(30 * s), config.NEON_CYAN)

        dx, dy = fx.shake_offset()
        if dx or dy:
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            frame[:] = cv2.warpAffine(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)

    def _draw_hud(self, frame):
        app, s, w, h = self.app, self.app.scale, self.app.w, self.app.h
        # Score (top-right, huge)
        sc = 1.0 + 0.22 * self.score_pop
        draw_text(frame, str(self.score), (w - int(30 * s), int(62 * s)), int(88 * s), config.WHITE,
                  anchor="right", scale=sc, style="black", gradient=((120, 255, 255), (30, 170, 255)),
                  outline_w=int(6 * s))
        best = max(app.leaderboard.best(), self.score)
        draw_text(frame, f"BEST {best}", (w - int(32 * s), int(126 * s)), int(26 * s), config.GOLD, anchor="right")

        # Lives / timer (top-left)
        if self.mode == MODE_SURVIVAL:
            icon = app.life_icon
            gray = app.life_icon_lost
            size = icon.shape[0]
            for i in range(config.START_LIVES):
                cx = int(28 * s) + size // 2 + i * int(size * 1.05)
                cy = int(20 * s) + size // 2
                if i < self.lives:
                    blit_center(frame, icon, cx, cy)
                else:
                    blit_center(frame, gray, cx, cy)
                    r = int(size * 0.32)
                    cv2.line(frame, (cx - r, cy - r), (cx + r, cy + r), config.NEON_RED, max(3, int(7 * s)), cv2.LINE_AA)
                    cv2.line(frame, (cx - r, cy + r), (cx + r, cy - r), config.NEON_RED, max(3, int(7 * s)), cv2.LINE_AA)
            draw_text(frame, "SURVIVAL", (int(30 * s), int(20 * s) + size + int(18 * s)), int(22 * s),
                      config.NEON_CYAN, anchor="left")
        else:
            tl = self.time_left
            secs = int(math.ceil(tl))
            urgent = tl <= 10
            col = config.NEON_RED if urgent else config.WHITE
            sc = 1.0 + (0.15 * (1 - (tl % 1.0)) if urgent else 0)
            draw_text(frame, f"{secs // 60}:{secs % 60:02d}", (int(30 * s), int(62 * s)), int(80 * s), col,
                      anchor="left", style="black", scale=sc, outline_w=int(6 * s))
            draw_text(frame, "TIME ATTACK", (int(32 * s), int(124 * s)), int(22 * s), config.NEON_CYAN, anchor="left")
            # progress bar
            bw = int(220 * s)
            x0, y0 = int(32 * s), int(144 * s)
            cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + int(8 * s)), (60, 40, 50), -1)
            cv2.rectangle(frame, (x0, y0), (x0 + int(bw * tl / config.TIMER_MODE_SECONDS), y0 + int(8 * s)),
                          col if urgent else config.NEON_CYAN, -1)

        if not app.hand_present and not app.mouse_blade:
            draw_hand_status(app, frame)


# ---------------------------------------------------------------------------

class GameOverState(State):
    text_input = True
    MAX_NAME = 12

    def __init__(self, app, mode, score, reason, sliced, best_combo):
        super().__init__(app)
        self.mode, self.score, self.reason = mode, score, reason
        self.sliced, self.best_combo = sliced, best_combo
        prev_best = app.leaderboard.best()
        self.new_high = score > 0 and score > prev_best
        self.name = ""
        self.entry_id = None
        self.idle = 0.0
        # Save straight away under a placeholder name so a crash or a closed
        # window during name entry can't lose the score.
        if score > 0:
            self.entry_id = app.leaderboard.add("PLAYER", score, mode)
        self.rank = app.leaderboard.rank_of(self.entry_id) if self.entry_id else None

    def enter(self):
        self.app.blade.reset()
        self.app.audio.play("highscore" if self.new_high else "gameover")

    def _finish(self, keep=True):
        lb = self.app.leaderboard
        highlight = None
        if self.entry_id:
            if keep:
                lb.rename(self.entry_id, self.name or "PLAYER")
                highlight = self.entry_id
            else:
                lb.remove(self.entry_id)
        self.app.audio.play("select")
        self.app.change(LeaderboardState(self.app, highlight=highlight))

    def on_key(self, key):
        self.idle = 0.0
        if self.entry_id is None:
            if key in ("space", "enter", "esc"):
                self._finish()
            return
        if key == "enter":
            self._finish(keep=True)
        elif key == "esc":
            self._finish(keep=False)
        elif key == "backspace":
            self.name = self.name[:-1]
        elif key == "space":
            if self.name and len(self.name) < self.MAX_NAME:
                self.name += " "
        elif len(key) == 1 and (key.isalnum() or key in "-_.!") and len(self.name) < self.MAX_NAME:
            self.name += key.upper()
            self.app.audio.play("tick", 0.6)

    def update(self, dt, now):
        super().update(dt, now)
        self.idle += dt
        if self.idle > config.GAMEOVER_IDLE_AUTOSAVE:
            self._finish(keep=True)

    def draw(self, frame, now):
        app, s, w, h = self.app, self.app.scale, self.app.w, self.app.h
        darken(frame, 0.45)
        pw, ph = int(820 * s), int(545 * s)
        px, py = w // 2 - pw // 2, int(118 * s)
        draw_panel(frame, px, py, pw, ph, fill=(35, 15, 30), fill_alpha=215,
                   border=config.NEON_PINK, border_w=3, radius=int(28 * s))

        title = {"bomb": "KABOOM!", "lives": "GAME OVER", "time": "TIME'S UP!"}.get(self.reason, "GAME OVER")
        draw_text(frame, title, (w // 2, py + int(62 * s)), int(64 * s), config.WHITE, style="black",
                  gradient=((120, 255, 255), (40, 120, 255)))
        draw_text(frame, MODE_LABELS[self.mode], (w // 2, py + int(112 * s)), int(24 * s), config.NEON_CYAN)

        pop = min(1.0, self.t * 3)
        draw_text(frame, str(self.score), (w // 2, py + int(195 * s)), int(110 * s), config.WHITE, style="black",
                  scale=0.5 + 0.5 * pop + 0.05 * math.sin(self.t * 4),
                  gradient=((140, 255, 255), (30, 170, 255)), outline_w=int(7 * s))
        draw_text(frame, f"FRUITS {self.sliced}     BEST COMBO {self.best_combo or '-'}",
                  (w // 2, py + int(270 * s)), int(24 * s), (220, 220, 220))

        if self.new_high:
            hue = int(self.t * 90) % 180
            c = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            sc = 1.0 + 0.08 * math.sin(self.t * 8)
            draw_text(frame, "NEW HIGH SCORE!", (w // 2, py - int(44 * s)), int(54 * s),
                      tuple(int(v) for v in c), style="black", scale=sc)
        elif self.rank:
            draw_text(frame, f"You placed #{self.rank}!", (w // 2, py - int(40 * s)), int(36 * s),
                      config.NEON_YELLOW, style="black")

        if self.entry_id is None:
            draw_text(frame, "No points this time - try again!", (w // 2, py + int(360 * s)), int(32 * s),
                      config.WHITE)
            if blink(self.t):
                draw_text(frame, "PRESS SPACE TO CONTINUE", (w // 2, py + int(470 * s)), int(32 * s),
                          config.NEON_YELLOW, style="black")
            return

        draw_text(frame, "ENTER YOUR NAME", (w // 2, py + int(325 * s)), int(30 * s), config.NEON_YELLOW,
                  style="black")
        bw, bh = int(520 * s), int(78 * s)
        bx, by = w // 2 - bw // 2, py + int(360 * s)
        draw_panel(frame, bx, by, bw, bh, fill=(15, 5, 10), fill_alpha=230, border=config.NEON_CYAN,
                   border_w=3, radius=int(14 * s))
        shown = self.name
        cursor = "_" if blink(self.t, 0.8) and len(self.name) < self.MAX_NAME else " "
        tw, _ = draw_text(frame, shown + cursor if shown else cursor, (w // 2, by + bh // 2), int(48 * s),
                          config.WHITE, style="black")
        if not self.name and not blink(self.t, 0.8):
            draw_text(frame, "type on the keyboard", (w // 2, by + bh // 2), int(26 * s), (130, 130, 130))
        draw_text(frame, "ENTER save    BACKSPACE delete    ESC skip", (w // 2, py + int(490 * s)),
                  int(22 * s), (200, 200, 200))


# ---------------------------------------------------------------------------

class LeaderboardState(State):
    def __init__(self, app, highlight=None, attract=False):
        super().__init__(app)
        self.highlight = highlight
        self.attract = attract
        self.idle = 0.0

    def enter(self):
        self.app.blade.reset()

    def on_key(self, key):
        if key in ("space", "enter", "esc", "l"):
            self.app.audio.play("select")
            self.app.change(MenuState(self.app))

    def update(self, dt, now):
        super().update(dt, now)
        self.idle += dt
        limit = config.LEADERBOARD_IDLE_TO_MENU * (1 if self.attract else 2)
        if self.idle > limit and not self.app.hand_present:
            self.app.change(MenuState(self.app))

    def draw(self, frame, now):
        app, s, w, h = self.app, self.app.scale, self.app.w, self.app.h
        darken(frame, 0.35)
        app.draw_blade(frame, now)
        draw_text(frame, "LEADERBOARD", (w // 2, int(60 * s)), int(68 * s), config.WHITE, style="black",
                  gradient=((120, 255, 255), (40, 120, 255)))
        pw, ph = int(860 * s), int(520 * s)
        px, py = w // 2 - pw // 2, int(112 * s)
        draw_panel(frame, px, py, pw, ph, fill=(35, 15, 30), fill_alpha=205, border=config.NEON_CYAN,
                   border_w=3, radius=int(26 * s))

        entries = app.leaderboard.top(config.LEADERBOARD_SHOW)
        row_h = int(46 * s)
        y0 = py + int(30 * s)
        cols = (px + int(70 * s), px + int(150 * s), px + pw - int(230 * s), px + pw - int(40 * s))
        hdr = int(20 * s)
        draw_text(frame, "RANK", (cols[0], y0), hdr, (170, 170, 170))
        draw_text(frame, "NAME", (cols[1], y0), hdr, (170, 170, 170), anchor="left")
        draw_text(frame, "MODE", (cols[2], y0), hdr, (170, 170, 170))
        draw_text(frame, "SCORE", (cols[3], y0), hdr, (170, 170, 170), anchor="right")
        if not entries:
            draw_text(frame, "No scores yet - be the first ninja!", (w // 2, py + ph // 2), int(34 * s),
                      config.NEON_YELLOW)
        medal = {0: config.GOLD, 1: config.SILVER, 2: config.BRONZE}
        for i, e in enumerate(entries):
            cy = y0 + int(42 * s) + i * row_h
            is_new = e["id"] == self.highlight
            if is_new:
                pulse = 0.5 + 0.5 * math.sin(self.t * 6)
                draw_panel(frame, px + int(18 * s), cy - row_h // 2 + 2, pw - int(36 * s), row_h - 4,
                           fill=(90, 30, 110), fill_alpha=int(150 + 80 * pulse), border=config.NEON_PINK,
                           border_w=2, radius=int(12 * s))
            elif i % 2 == 0:
                draw_panel(frame, px + int(18 * s), cy - row_h // 2 + 2, pw - int(36 * s), row_h - 4,
                           fill=(60, 35, 55), fill_alpha=90, radius=int(12 * s))
            col = medal.get(i, config.WHITE)
            if is_new:
                col = config.NEON_YELLOW
            size = int(30 * s)
            draw_text(frame, f"#{i + 1}", (cols[0], cy), size, col, style="black")
            draw_text(frame, e["name"], (cols[1], cy), size, config.WHITE if not is_new else config.NEON_YELLOW,
                      anchor="left")
            draw_text(frame, "60s" if e.get("mode") == MODE_TIMER else "SURV", (cols[2], cy), int(20 * s),
                      (190, 190, 190))
            draw_text(frame, str(e["score"]), (cols[3], cy), size, col, anchor="right", style="black")

        if self.highlight and app.leaderboard.rank_of(self.highlight) and \
                app.leaderboard.rank_of(self.highlight) > config.LEADERBOARD_SHOW:
            draw_text(frame, f"Your score is #{app.leaderboard.rank_of(self.highlight)} - so close!",
                      (w // 2, py + ph + int(24 * s)), int(26 * s), config.NEON_YELLOW)
        if blink(self.t, 1.1):
            draw_text(frame, "PRESS SPACE TO RETURN TO MENU", (w // 2, h - int(36 * s)), int(32 * s),
                      config.NEON_YELLOW, style="black")
