"""Turns an uploaded photo into a wardrobe item: cut-out, embedding, slot/type, colour, style."""
from __future__ import annotations

import io
import threading

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .encoder import Encoder
from .taxonomy import SLOT_ID

# fine-grained garment types -> slot (zero-shot classified with the fashion encoder)
TYPES = {
    "t-shirt": "top", "shirt": "top", "blouse": "top", "sweater": "top", "hoodie": "top", "sweatshirt": "top",
    "tank top": "top", "polo shirt": "top", "crop top": "top", "turtleneck": "top",
    "jeans": "bottom", "trousers": "bottom", "shorts": "bottom", "skirt": "bottom", "leggings": "bottom",
    "sweatpants": "bottom", "chinos": "bottom", "cargo pants": "bottom",
    "dress": "dress", "jumpsuit": "dress", "gown": "dress", "romper": "dress",
    "jacket": "outerwear", "coat": "outerwear", "blazer": "outerwear", "cardigan": "outerwear",
    "vest": "outerwear", "denim jacket": "outerwear", "leather jacket": "outerwear", "puffer jacket": "outerwear",
    "sneakers": "shoes", "boots": "shoes", "high heels": "shoes", "sandals": "shoes", "loafers": "shoes",
    "flat shoes": "shoes", "dress shoes": "shoes", "slippers": "shoes",
    "handbag": "bag", "backpack": "bag", "tote bag": "bag", "clutch bag": "bag", "crossbody bag": "bag",
    "necklace": "jewelry", "earrings": "jewelry", "bracelet": "jewelry", "ring": "jewelry",
    "wrist watch": "watch",
    "hat": "hat", "baseball cap": "hat", "beanie": "hat",
    "sunglasses": "eyewear", "eyeglasses": "eyewear",
    "scarf": "scarf", "belt": "belt",
    "necktie": "accessory", "gloves": "accessory", "socks": "accessory",
}
TYPE_NAMES = list(TYPES)

STYLES = ["casual", "smart casual", "formal", "sporty", "party", "bohemian", "streetwear", "classic"]

# name -> (rgb, hue family or None for neutrals)
PALETTE = {
    "black": ((20, 20, 22), None), "charcoal": ((60, 62, 66), None), "gray": ((140, 140, 142), None),
    "white": ((245, 245, 243), None), "cream": ((238, 228, 205), None), "beige": ((210, 190, 160), None),
    "tan": ((190, 150, 105), None), "brown": ((110, 72, 45), None), "navy": ((28, 38, 78), None),
    "khaki": ((150, 140, 100), None),
    "blue": ((45, 95, 190), "blue"), "light blue": ((150, 190, 225), "blue"), "teal": ((20, 130, 130), "green"),
    "green": ((40, 140, 70), "green"), "olive": ((95, 105, 50), "green"), "mint": ((170, 225, 195), "green"),
    "yellow": ((240, 215, 60), "yellow"), "mustard": ((200, 155, 40), "yellow"),
    "orange": ((235, 130, 40), "orange"), "coral": ((240, 125, 105), "orange"),
    "red": ((200, 35, 40), "red"), "burgundy": ((110, 25, 45), "red"),
    "pink": ((240, 150, 180), "pink"), "blush": ((240, 205, 205), "pink"), "hot pink": ((225, 50, 130), "pink"),
    "purple": ((115, 60, 150), "purple"), "lavender": ((190, 170, 225), "purple"),
}
_PAL_NAMES = list(PALETTE)
_PAL_LAB = cv2.cvtColor(np.array([[PALETTE[n][0] for n in _PAL_NAMES]], np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def hue_family(color: str, slot: str):
    fam = PALETTE.get(color, (None, None))[1]
    if fam == "blue" and slot == "bottom":  # denim behaves as a neutral
        return None
    return fam


class Vision:
    def __init__(self, enc: Encoder):
        self.enc = enc
        self.type_emb = enc.embed_text([f"a photo of a {t}, a fashion product" for t in TYPE_NAMES])
        self.style_emb = enc.embed_text([f"{s} style clothing" for s in STYLES])
        self._rembg = None
        self._rembg_lock = threading.Lock()

    # ---------- background removal ----------
    def cutout(self, im: Image.Image) -> Image.Image:
        """RGBA cut-out cropped to the garment. Falls back to the plain photo if matting fails."""
        try:
            from rembg import new_session, remove
            with self._rembg_lock:
                if self._rembg is None:
                    self._rembg = new_session("silueta")
                    if self.enc.device == "cpu":  # small hosts: no memory arena (~0.4 GB less RAM)
                        import onnxruntime as ort
                        o = ort.SessionOptions()
                        o.enable_cpu_mem_arena = False
                        o.enable_mem_pattern = False
                        o.intra_op_num_threads = 2
                        self._rembg.inner_session = ort.InferenceSession(
                            self._rembg.inner_session._model_path, sess_options=o,
                            providers=["CPUExecutionProvider"])
                out = remove(im, session=self._rembg, post_process_mask=True)
            a = np.array(out)[..., 3]
            frac = (a > 128).mean()
            if not 0.03 < frac < 0.97:
                raise ValueError("degenerate mask")
            ys, xs = np.where(a > 128)
            out = out.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            return out
        except Exception:
            return im.convert("RGBA")

    @staticmethod
    def on_white(rgba: Image.Image, pad=0.08) -> Image.Image:
        """Square white canvas - the look of the product shots the model was trained on."""
        w, h = rgba.size
        side = int(max(w, h) * (1 + 2 * pad))
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(rgba, ((side - w) // 2, (side - h) // 2), rgba)
        return canvas

    # ---------- colour ----------
    @staticmethod
    def colors(rgba: Image.Image):
        arr = np.array(rgba.resize((96, 96)))
        px = arr[..., :3][arr[..., 3] > 128]
        if len(px) > 0.97 * 96 * 96:  # no mask: use the centre, drop near-white background
            c = arr[24:72, 24:72, :3].reshape(-1, 3)
            keep = c.min(1) < 235
            px = c[keep] if keep.sum() > 200 else c
        if len(px) < 10:
            return [("gray", "#8c8c8e")]
        lab = cv2.cvtColor(px[None].astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
        k = min(3, len(lab))
        _, labels, centers = cv2.kmeans(lab, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                        3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k)
        out = []
        for i in np.argsort(-counts):
            if counts[i] / counts.sum() < 0.18 and out:
                continue
            name = _PAL_NAMES[int(np.linalg.norm(_PAL_LAB - centers[i], axis=1).argmin())]
            rgb = cv2.cvtColor(centers[i][None, None].astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]
            if name not in [n for n, _ in out]:
                out.append((name, "#%02x%02x%02x" % tuple(int(v) for v in rgb)))
        return out

    # ---------- full analysis ----------
    def analyse(self, data: bytes) -> dict:
        im = Image.open(io.BytesIO(data))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((1024, 1024))
        rgba = self.cutout(im)
        emb = self.enc.embed_images([self.on_white(rgba)])[0]
        t = int((emb @ self.type_emb.T).argmax())
        s = int((emb @ self.style_emb.T).argmax())
        cols = self.colors(rgba)
        rgba.thumbnail((480, 480))
        buf = io.BytesIO()
        rgba.save(buf, "PNG", optimize=True)
        typ = TYPE_NAMES[t]
        return dict(
            emb=emb.cpu().numpy().astype(np.float32), png=buf.getvalue(),
            type=typ, slot=TYPES[typ], style=STYLES[s],
            color=cols[0][0], colors=[dict(name=n, hex=h) for n, h in cols],
            name=f"{cols[0][0]} {typ}",
        )
