"""Per-user preference model that learns from likes, dislikes and reasons (US-013).

The outfit score is a weighted sum of interpretable features. Feedback does
three things:
  * nudges the feature weights (online logistic regression step),
  * moves affinities for the specific items, colour pairs and styles involved,
  * applies the stated reason directly (e.g. "too formal" lowers the formality
    target for that occasion).
"""
from __future__ import annotations

import math
from itertools import combinations

from .schema import Occasion, Outfit, Reason

FEATURES = ("harmony", "formality", "weather", "style", "color_pref", "pattern", "novelty")
DEFAULT_WEIGHTS = {"harmony": .22, "formality": .26, "weather": .16, "style": .12,
                   "color_pref": .10, "pattern": .06, "novelty": .08}


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def color_pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


class PreferenceModel:
    def __init__(self):
        self.weights = dict(DEFAULT_WEIGHTS)
        self.item_affinity = {}        # item id -> [-1, 0.3]
        self.color_pair = {}           # "a|b" -> penalty/bonus
        self.style_affinity = {}       # style -> [-0.5, 0.5]
        self.formality_offset = {}     # occasion value -> [-0.3, 0.3]
        self.warmth_offset = 0.0       # [-0.3, 0.3]
        self.n_feedback = 0

    # ---- scoring -------------------------------------------------------
    def score(self, features: dict, items: list, challenge: bool = False) -> float:
        w = dict(self.weights)
        if challenge:                  # style challenges favour rarely worn pieces
            w["novelty"] *= 3.0
        total = sum(w.values())
        base = sum(w[f] * features.get(f, 0.5) for f in FEATURES) / total

        aff = [self.item_affinity.get(i.id, 0.0) for i in items]
        adj = _clamp(sum(aff) / len(aff), -0.3, 0.15) if aff else 0.0
        if any(a <= -0.5 for a in aff):            # an item the user said they dislike
            adj -= 0.3
        names = sorted({i.main_color.name for i in items if i.main_color})
        pair = sum(self.color_pair.get(color_pair_key(a, b), 0.0)
                   for a, b in combinations(names, 2))
        return base + adj + _clamp(pair, -0.25, 0.1)

    # ---- learning ------------------------------------------------------
    def update(self, outfit: Outfit, liked: bool, occasion=None,
               reason: Reason = None, item_id: str = None, lr: float = 0.15):
        self.n_feedback += 1
        y = 1.0 if liked else 0.0
        p = 1.0 / (1.0 + math.exp(-8.0 * (outfit.score - 0.55)))
        specific = reason not in (None, Reason.OTHER)

        # A stated reason explains the rejection, so the generic updates are
        # softened to avoid blaming everything else in the outfit.
        scale = 0.3 if (specific and not liked) else 1.0
        for f in FEATURES:
            self.weights[f] = max(0.02, self.weights[f]
                                  + scale * lr * (y - p) * (outfit.features.get(f, 0.5) - 0.5))
        total = sum(self.weights.values())
        self.weights = {f: v / total for f, v in self.weights.items()}

        step = (0.04 if liked else -0.04) * scale
        for i in outfit.items:
            self.item_affinity[i.id] = _clamp(self.item_affinity.get(i.id, 0.0) + step, -1, 0.3)

        if liked or reason is None:
            return
        reason = Reason(reason)
        occ = Occasion(occasion).value if occasion else None
        if reason is Reason.DISLIKE_ITEM and item_id:
            self.item_affinity[item_id] = -1.0
        elif reason is Reason.COLORS:
            names = sorted({i.main_color.name for i in outfit.items if i.main_color})
            for a, b in combinations(names, 2):
                k = color_pair_key(a, b)
                self.color_pair[k] = _clamp(self.color_pair.get(k, 0.0) - 0.08, -0.4, 0.1)
        elif reason in (Reason.TOO_FORMAL, Reason.TOO_CASUAL) and occ:
            d = -0.08 if reason is Reason.TOO_FORMAL else 0.08
            self.formality_offset[occ] = _clamp(self.formality_offset.get(occ, 0.0) + d, -.3, .3)
        elif reason in (Reason.TOO_WARM, Reason.TOO_COLD):
            d = -0.08 if reason is Reason.TOO_WARM else 0.08
            self.warmth_offset = _clamp(self.warmth_offset + d, -.3, .3)
        elif reason is Reason.NOT_MY_STYLE:
            counts = {}
            for i in outfit.items:
                for s in i.styles:
                    counts[s] = counts.get(s, 0) + 1
            if counts:
                s = max(sorted(counts), key=counts.get)
                self.style_affinity[s] = _clamp(self.style_affinity.get(s, 0.0) - 0.15, -.5, .5)

    # ---- persistence ---------------------------------------------------
    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "PreferenceModel":
        m = cls()
        m.__dict__.update(d)
        return m
