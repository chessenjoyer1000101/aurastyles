"""Wardrobe photo -> WardrobeItem (US-003).

Category and pattern come from zero-shot CLIP: the image is compared with one
text prompt per garment category, so no labelled training data is needed and
adding a category is a one-line change in knowledge.py. Colours come from
colors.extract_palette and need no model.

The classifier only ever compares images with garment prompts. It does not
estimate anything about a person in the photo (US-015).
"""
from __future__ import annotations

import importlib.util
import uuid

import numpy as np

from .colors import extract_palette, pattern_from_palette
from .knowledge import CATEGORIES, PATTERNS, build_item
from .schema import WardrobeItem

CATEGORY_PROMPT = "a photo of a {}, a clothing item"
PATTERN_PROMPT = "a photo of clothing with a {} pattern"
PATTERN_WORDS = {"solid": "plain solid colour, no", "graphic": "graphic print"}


class ClipBackend:
    """Zero-shot image/text matcher built on open_clip. Loaded on first use.

    Defaults to general-purpose ViT-B/32. A fashion-tuned checkpoint such as
    model_name="hf-hub:Marqo/marqo-fashionSigLIP" (pretrained=None) usually
    separates garment categories better.
    """

    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
        self.model_name, self.pretrained, self.device = model_name, pretrained, device
        self._model = None
        self._text_cache = {}

    @staticmethod
    def available() -> bool:
        return (importlib.util.find_spec("open_clip") is not None
                and importlib.util.find_spec("torch") is not None)

    def _load(self):
        if self._model is not None:
            return
        import open_clip
        import torch
        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"pretrained": self.pretrained} if self.pretrained else {}
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, **kw)
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def _text_features(self, prompts: tuple):
        if prompts not in self._text_cache:
            with self._torch.no_grad():
                t = self._model.encode_text(self._tokenizer(list(prompts)).to(self.device))
                self._text_cache[prompts] = t / t.norm(dim=-1, keepdim=True)
        return self._text_cache[prompts]

    def zero_shot(self, image, prompts: list) -> np.ndarray:
        """Probability of each prompt for the image; sums to 1."""
        self._load()
        with self._torch.no_grad():
            x = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
            f = self._model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            logits = 100.0 * f @ self._text_features(tuple(prompts)).T
            return logits.softmax(dim=-1)[0].cpu().numpy()


class ItemClassifier:
    """Turns a photo of one garment into a WardrobeItem.

    backend: "auto" uses ClipBackend when open_clip and torch are installed,
    None disables category recognition, or pass any object with a
    zero_shot(image, prompts) -> probabilities method.

    An item is flagged needs_review when the top category is not confident
    enough or is too close to the runner-up; the caller should then show
    `alternatives` and let the user choose.
    """

    def __init__(self, backend="auto", min_confidence=0.45, min_margin=0.15):
        if backend == "auto":
            backend = ClipBackend() if ClipBackend.available() else None
        self.backend = backend
        self.min_confidence, self.min_margin = min_confidence, min_margin
        self.categories = list(CATEGORIES)

    def classify(self, image, item_id=None) -> WardrobeItem:
        from PIL import Image
        path = None
        if not isinstance(image, Image.Image):
            path = str(image)
            image = Image.open(path)
        item_id = item_id or uuid.uuid4().hex[:8]
        palette = extract_palette(image)

        if self.backend is None:
            pattern, _ = pattern_from_palette(palette)
            item = build_item(item_id, "unknown", palette, pattern)
            item.image_path = path
            return item

        probs = self.backend.zero_shot(
            image, [CATEGORY_PROMPT.format(c) for c in self.categories])
        order = np.argsort(-probs)
        top, second = float(probs[order[0]]), float(probs[order[1]])
        pprobs = self.backend.zero_shot(
            image, [PATTERN_PROMPT.format(PATTERN_WORDS.get(p, p)) for p in PATTERNS])
        pattern = PATTERNS[int(np.argmax(pprobs))]

        item = build_item(item_id, self.categories[order[0]], palette, pattern)
        item.confidence = round(top, 3)
        item.needs_review = top < self.min_confidence or (top - second) < self.min_margin
        item.alternatives = [(self.categories[i], round(float(probs[i]), 3))
                             for i in order[:3]]
        item.image_path = path
        return item

    def classify_many(self, images) -> list:
        return [self.classify(im) for im in images]
