"""Colour perception: palette extraction from a garment photo, naming, and harmony."""
from __future__ import annotations

import colorsys

import numpy as np

from .schema import ColorInfo

COLOR_RGB = {
    "black": (20, 20, 22), "white": (245, 245, 245), "gray": (128, 128, 128),
    "navy": (25, 35, 80), "beige": (215, 195, 160), "cream": (245, 238, 215),
    "brown": (110, 70, 40), "red": (200, 30, 40), "burgundy": (110, 20, 40),
    "pink": (240, 150, 180), "orange": (240, 130, 30), "yellow": (240, 210, 50),
    "green": (40, 150, 70), "olive": (100, 105, 45), "teal": (20, 140, 140),
    "blue": (40, 90, 200), "light blue": (150, 190, 235), "purple": (120, 60, 170),
}
NEUTRALS = {"black", "white", "gray", "navy", "beige", "cream", "brown"}


def hue_of(rgb) -> float:
    r, g, b = (c / 255.0 for c in rgb)
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360.0


def name_color(rgb) -> str:
    """Map an RGB value to one of the names in COLOR_RGB."""
    r, g, b = (float(c) / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360.0
    if v < 0.18:
        return "black"
    if s < 0.10:
        return "white" if v > 0.85 else "gray"
    if s < 0.20 and v > 0.88 and 20 <= h <= 70:
        return "cream"
    if s < 0.35 and v > 0.60 and 20 <= h <= 60:
        return "beige"
    if (h >= 335 or h < 15) and v < 0.50:
        return "burgundy"
    if 10 <= h <= 45 and v < 0.60 and s >= 0.30:
        return "brown"
    if 200 <= h <= 260 and v < 0.42:
        return "navy"
    if 50 <= h <= 100 and v < 0.55 and s >= 0.25:
        return "olive"
    if h < 15 or h >= 345:
        return "pink" if (s < 0.50 and v > 0.80) else "red"
    if h < 40:
        return "orange"
    if h < 65:
        return "yellow"
    if h < 165:
        return "green"
    if h < 195:
        return "teal"
    if h < 255:
        return "light blue" if (v > 0.75 and s < 0.50) else "blue"
    if h < 290:
        return "purple"
    return "pink"


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (N,3) in 0..255 -> CIELAB (N,3), D65."""
    c = rgb.astype(np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                     200 * (f[:, 1] - f[:, 2])], axis=1)


def _kmeans(x: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    """Deterministic k-means; returns a label per row."""
    order = np.argsort(x[:, 0], kind="stable")
    idx = order[np.linspace(0, len(x) - 1, k).astype(int)]
    centers = x[idx].copy()
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        labels = d.argmin(1)
        for j in range(k):
            if (labels == j).any():
                centers[j] = x[labels == j].mean(0)
    return labels


def extract_palette(image, k: int = 4, max_colors: int = 3) -> list:
    """Dominant garment colours of a PIL image, most prominent first.

    A uniform backdrop (flat-lay or product-style photo) is detected from the
    image border and excluded. Transparent pixels are ignored.
    """
    im = image.convert("RGBA")
    im.thumbnail((96, 96))
    a = np.asarray(im).astype(np.float32)
    rgb, alpha = a[..., :3], a[..., 3]
    h, w = rgb.shape[:2]
    lab = rgb2lab(rgb.reshape(-1, 3)).reshape(h, w, 3)
    mask = alpha > 32

    b = max(2, int(0.06 * min(h, w)))
    border = np.zeros((h, w), bool)
    border[:b] = border[-b:] = True
    border[:, :b] = border[:, -b:] = True
    bl = lab[border & mask]
    if len(bl):
        med = np.median(bl, 0)
        spread = np.linalg.norm(bl - med, axis=1).mean()
        if spread < 10:                                         # backdrop is uniform
            # the flatter the backdrop, the closer a garment colour may sit to it
            fg = np.linalg.norm(lab - med, axis=2) > max(6.0, 3.0 * spread + 4.0)
            if (fg & mask).mean() > 0.05:                       # garment differs from it
                mask &= fg

    px_lab, px_rgb = lab[mask], rgb[mask]
    if len(px_lab) < 20:
        px_lab, px_rgb = lab.reshape(-1, 3), rgb.reshape(-1, 3)
    labels = _kmeans(px_lab, min(k, len(px_lab)))

    agg = {}  # colour name -> [pixel count, rgb sum]
    for j in np.unique(labels):
        sel = labels == j
        mean = px_rgb[sel].mean(0)
        entry = agg.setdefault(name_color(mean), [0, np.zeros(3)])
        entry[0] += int(sel.sum())
        entry[1] += mean * sel.sum()
    total = sum(v[0] for v in agg.values())
    out = [ColorInfo(n, tuple(int(round(c)) for c in v[1] / v[0]), round(v[0] / total, 3))
           for n, v in agg.items()]
    out.sort(key=lambda c: -c.share)
    kept = [c for c in out if c.share >= 0.08] or out[:1]
    return kept[:max_colors]


def pattern_from_palette(palette: list) -> tuple:
    """Fallback pattern guess when no vision model is available: (pattern, confidence)."""
    if not palette or palette[0].share >= 0.80:
        return "solid", 0.6
    return "patterned", 0.4


def _pair(d: float) -> float:
    if d < 30:
        return 1.0      # same hue family
    if d < 60:
        return 0.8      # analogous
    if d < 100:
        return 0.4      # neither close nor contrasting
    if d < 150:
        return 0.6      # triadic-ish
    return 0.8          # complementary


def harmony_score(colors: list) -> float:
    """0..1 score for how well a set of main colours sits together.

    Neutrals combine with anything. Chromatic colours are scored by their hue
    relationships, with a penalty once more than three hue families appear.
    """
    hues = [hue_of(c.rgb) for c in colors if c is not None and c.name not in NEUTRALS]
    if not hues:
        return 0.80
    if len(hues) == 1:
        return 0.95
    pairs = []
    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            d = abs(hues[i] - hues[j])
            pairs.append(_pair(min(d, 360 - d)))
    families = len({int(h // 30) for h in hues})
    return float(max(0.0, np.mean(pairs) - 0.15 * max(0, families - 3)))
