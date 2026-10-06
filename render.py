"""Drawing helpers: cached PIL text sprites with outline + shadow, fast RGBA
blitting onto BGR frames, translucent panels and the vignette overlay."""

import os
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

_FONT_CANDIDATES = {
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Rounded Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Rounded Bold.ttf",
        "C:/Windows/Fonts/ARLRDBD.TTF",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
    "black": [
        "/System/Library/Fonts/Supplemental/Arial Black.ttf",
        "/Library/Fonts/Arial Black.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ],
}

_font_paths = {}
_font_cache = {}


def _font_path(style):
    if style not in _font_paths:
        _font_paths[style] = next(
            (p for p in _FONT_CANDIDATES.get(style, []) if os.path.exists(p)), None)
    return _font_paths[style]


def get_font(size, style="bold"):
    key = (size, style)
    if key not in _font_cache:
        path = _font_path(style) or _font_path("bold")
        try:
            _font_cache[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
        except (OSError, TypeError):
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def bgr_to_rgb(c):
    return (c[2], c[1], c[0])


# --- Text ------------------------------------------------------------------

_text_cache = OrderedDict()
_TEXT_CACHE_MAX = 400


def text_sprite(text, size, color=(255, 255, 255), outline=(20, 10, 15), outline_w=None,
                shadow=True, style="bold", gradient=None):
    """Render text to a BGRA uint8 array (cached). `gradient` is an optional
    (top_bgr, bottom_bgr) pair that replaces the flat fill color."""
    size = max(6, int(size))
    if outline_w is None:
        outline_w = max(2, size // 14)
    key = (text, size, color, outline, outline_w, shadow, style, gradient)
    sprite = _text_cache.get(key)
    if sprite is not None:
        _text_cache.move_to_end(key)
        return sprite

    font = get_font(size, style)
    pad = outline_w + (size // 8 if shadow else 0) + 4
    bbox = font.getbbox(text, stroke_width=outline_w)
    w = bbox[2] - bbox[0] + pad * 2
    h = bbox[3] - bbox[1] + pad * 2
    origin = (pad - bbox[0], pad - bbox[1])
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    if shadow:
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        off = max(2, size // 16)
        ImageDraw.Draw(sh).text((origin[0] + off, origin[1] + off), text, font=font,
                                fill=(0, 0, 0, 170), stroke_width=outline_w,
                                stroke_fill=(0, 0, 0, 170))
        sh = sh.filter(ImageFilter.GaussianBlur(max(1, size // 20)))
        img = Image.alpha_composite(img, sh)

    draw = ImageDraw.Draw(img)
    if outline is not None and outline_w > 0:
        draw.text(origin, text, font=font, fill=bgr_to_rgb(outline) + (255,),
                  stroke_width=outline_w, stroke_fill=bgr_to_rgb(outline) + (255,))
    if gradient is None:
        draw.text(origin, text, font=font, fill=bgr_to_rgb(color) + (255,))
    else:
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).text(origin, text, font=font, fill=255)
        top, bottom = np.array(bgr_to_rgb(gradient[0]), np.float32), np.array(bgr_to_rgb(gradient[1]), np.float32)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        rows = (top * (1 - t) + bottom * t).astype(np.uint8)
        grad = np.repeat(rows[:, None, :], w, axis=1)
        grad_img = Image.fromarray(np.dstack([grad, np.full((h, w), 255, np.uint8)]), "RGBA")
        img.paste(grad_img, (0, 0), mask)

    arr = np.array(img)
    sprite = np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])
    _text_cache[key] = sprite
    if len(_text_cache) > _TEXT_CACHE_MAX:
        _text_cache.popitem(last=False)
    return sprite


def draw_text(frame, text, pos, size, color=(255, 255, 255), anchor="center", alpha=1.0,
              scale=1.0, **kw):
    """Draw text at pos. anchor: center | left | right | topleft | topright | midtop."""
    sprite = text_sprite(text, size, color, **kw)
    if scale != 1.0 and scale > 0.01:
        sprite = cv2.resize(sprite, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w = sprite.shape[:2]
    x, y = pos
    if anchor == "center":
        x0, y0 = x - w // 2, y - h // 2
    elif anchor == "left":
        x0, y0 = x, y - h // 2
    elif anchor == "right":
        x0, y0 = x - w, y - h // 2
    elif anchor == "topleft":
        x0, y0 = x, y
    elif anchor == "topright":
        x0, y0 = x - w, y
    elif anchor == "midtop":
        x0, y0 = x - w // 2, y
    else:
        raise ValueError(anchor)
    blit(frame, sprite, int(x0), int(y0), alpha)
    return w, h


def text_width(text, size, style="bold"):
    return text_sprite(text, size, style=style).shape[1]


# --- Blitting --------------------------------------------------------------

def blit(frame, sprite, x0, y0, alpha=1.0):
    """Alpha-blend a BGRA sprite onto a BGR frame with its top-left at (x0, y0)."""
    if alpha <= 0.003:
        return
    fh, fw = frame.shape[:2]
    sh, sw = sprite.shape[:2]
    x1, y1 = max(x0, 0), max(y0, 0)
    x2, y2 = min(x0 + sw, fw), min(y0 + sh, fh)
    if x1 >= x2 or y1 >= y2:
        return
    src = sprite[y1 - y0:y2 - y0, x1 - x0:x2 - x0]
    dst = frame[y1:y2, x1:x2]
    a = src[:, :, 3:4].astype(np.float32) * (alpha / 255.0)
    dst[:] = (src[:, :, :3] * a + dst * (1.0 - a)).astype(np.uint8)


def blit_center(frame, sprite, cx, cy, alpha=1.0):
    h, w = sprite.shape[:2]
    blit(frame, sprite, int(round(cx - w / 2)), int(round(cy - h / 2)), alpha)


def rotate_sprite(sprite, angle_deg, scale=1.0):
    """Rotate a square-ish sprite around its center, expanding the canvas."""
    if abs(angle_deg) < 0.01 and scale == 1.0:
        return sprite
    h, w = sprite.shape[:2]
    side = int(np.ceil(np.hypot(w, h) * scale)) + 2
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, scale)
    m[0, 2] += side / 2 - w / 2
    m[1, 2] += side / 2 - h / 2
    return cv2.warpAffine(sprite, m, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# --- Panels ----------------------------------------------------------------

_panel_cache = {}


def panel_sprite(w, h, fill=(30, 15, 25), fill_alpha=190, border=None, border_w=3, radius=24):
    key = (w, h, fill, fill_alpha, border, border_w, radius)
    if key not in _panel_cache:
        ss = 2
        img = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w * ss - 1, h * ss - 1], radius=radius * ss,
                            fill=bgr_to_rgb(fill) + (fill_alpha,),
                            outline=(bgr_to_rgb(border) + (255,)) if border else None,
                            width=border_w * ss if border else 0)
        img = img.resize((w, h), Image.LANCZOS)
        arr = np.array(img)
        _panel_cache[key] = np.ascontiguousarray(arr[:, :, [2, 1, 0, 3]])
        if len(_panel_cache) > 64:
            _panel_cache.pop(next(iter(_panel_cache)))
    return _panel_cache[key]


def draw_panel(frame, x, y, w, h, alpha=1.0, **kw):
    blit(frame, panel_sprite(int(w), int(h), **kw), int(x), int(y), alpha)


def darken(frame, amount):
    """Darken the whole frame in place (amount 0..1)."""
    cv2.convertScaleAbs(frame, dst=frame, alpha=1.0 - amount)


def tint(frame, color, amount):
    """Blend a flat color over the whole frame in place."""
    if amount <= 0.003:
        return
    overlay = np.empty_like(frame)
    overlay[:] = color
    cv2.addWeighted(overlay, amount, frame, 1.0 - amount, 0, dst=frame)


# --- Vignette --------------------------------------------------------------

class Vignette:
    """Precomputed darkening mask: dims the camera feed (more at the edges) so
    neon UI and fruit pop against a busy background."""

    def __init__(self, w, h, center=0.78, edge=0.22):
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xs - w / 2) / (w / 2)
        ny = (ys - h / 2) / (h / 2)
        d = np.clip(np.sqrt(nx * nx * 0.8 + ny * ny * 1.0) / 1.25, 0, 1)
        mask = center + (edge - center) * (d ** 1.8)
        self.mask = np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        # cool, slightly purple grade for the arcade look
        self.grade = np.array([1.08, 0.94, 0.98], np.float32)

    def apply(self, frame):
        cv2.multiply(frame, self.mask, dst=frame, scale=1.0 / 255.0)
        cv2.transform(frame, np.diag(self.grade), dst=frame)
        return frame
