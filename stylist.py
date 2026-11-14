"""AuraStylist: the single entry point the rest of the product calls."""
from __future__ import annotations

import json

from .analysis import discover_style, find_gaps
from .feedback import PreferenceModel
from .knowledge import CATEGORIES, build_item
from .recommender import Recommender
from .schema import Context, Outfit, Recommendation, Reason, StyleProfile, WardrobeItem
from .vision import ItemClassifier


class AuraStylist:
    def __init__(self, profile: StyleProfile = None, classifier: ItemClassifier = None):
        self.profile = profile or StyleProfile()
        self.classifier = classifier
        self.wardrobe = {}                  # id -> WardrobeItem
        self.prefs = PreferenceModel()
        self.recommender = Recommender(self.prefs)
        self.saved = []                     # outfits the user kept (lists of item ids)
        self.rejected = set()               # outfit keys never to show again

    # ---- wardrobe (US-003) -------------------------------------------
    def add_photo(self, image) -> WardrobeItem:
        """Classify one photo and store the item. Check `needs_review` on the result."""
        if self.classifier is None:
            self.classifier = ItemClassifier()
        item = self.classifier.classify(image)
        self.wardrobe[item.id] = item
        return item

    def add_photos(self, images) -> list:
        return [self.add_photo(im) for im in images]

    def add_item(self, item: WardrobeItem) -> WardrobeItem:
        self.wardrobe[item.id] = item
        return item

    def correct_item(self, item_id: str, category: str = None, colors: list = None,
                     pattern: str = None) -> WardrobeItem:
        """Apply the user's correction; attributes are re-derived from the new category."""
        old = self.wardrobe[item_id]
        if category is not None and category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        item = build_item(item_id, category or old.category,
                          colors if colors is not None else old.colors,
                          pattern or old.pattern,
                          image_path=old.image_path, wear_count=old.wear_count)
        item.confidence, item.needs_review = 1.0, False
        self.wardrobe[item_id] = item
        return item

    def remove_item(self, item_id: str):
        self.wardrobe.pop(item_id, None)
        self.prefs.item_affinity.pop(item_id, None)
        self.saved = [s for s in self.saved if item_id not in s]

    def items_needing_review(self) -> list:
        return [i for i in self.wardrobe.values() if i.needs_review]

    # ---- recommendations (US-004..007, US-010) ------------------------
    def recommend(self, context: Context = None, k: int = 3) -> Recommendation:
        return self.recommender.recommend(list(self.wardrobe.values()), self.profile,
                                          context or Context(), k, exclude=self.rejected)

    def challenge(self, category: str = None, item_id: str = None,
                  context: Context = None, k: int = 3) -> Recommendation:
        """New looks around a rarely worn piece, skipping outfits already saved."""
        pool = [i for i in self.wardrobe.values() if i.slot is not None
                and (category is None or i.category == category)]
        if item_id is None and pool:
            item_id = min(pool, key=lambda i: (i.wear_count, i.id)).id
        saved_keys = {"+".join(sorted(s)) for s in self.saved}
        return self.recommender.recommend(
            list(self.wardrobe.values()), self.profile, context or Context(), k,
            must_include=item_id, exclude=self.rejected | saved_keys, challenge=True)

    # ---- feedback (US-013) --------------------------------------------
    def like(self, outfit: Outfit, context: Context = None, save: bool = True):
        self.prefs.update(outfit, True, (context or Context()).occasion)
        if save:
            self.saved.append(outfit.item_ids())
            for i in outfit.items:
                if i.id in self.wardrobe:
                    self.wardrobe[i.id].wear_count += 1

    def dislike(self, outfit: Outfit, context: Context = None, reason: Reason = None,
                item_id: str = None):
        self.prefs.update(outfit, False, (context or Context()).occasion, reason, item_id)
        self.rejected.add(outfit.key)

    # ---- insight (US-008, US-011) -------------------------------------
    def gaps(self) -> list:
        return find_gaps(list(self.wardrobe.values()), self.profile)

    def style_identity(self) -> list:
        liked = [Outfit([self.wardrobe[i] for i in ids if i in self.wardrobe])
                 for ids in self.saved]
        return discover_style(list(self.wardrobe.values()), self.profile, liked,
                              self.prefs.style_affinity)

    # ---- persistence and deletion (US-015) ----------------------------
    def to_dict(self) -> dict:
        return {"profile": self.profile.to_dict(),
                "wardrobe": [i.to_dict() for i in self.wardrobe.values()],
                "prefs": self.prefs.to_dict(), "saved": self.saved,
                "rejected": sorted(self.rejected)}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path: str, classifier: ItemClassifier = None) -> "AuraStylist":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s = cls(StyleProfile.from_dict(d["profile"]), classifier)
        s.wardrobe = {i["id"]: WardrobeItem.from_dict(i) for i in d["wardrobe"]}
        s.prefs = PreferenceModel.from_dict(d["prefs"])
        s.recommender = Recommender(s.prefs)
        s.saved, s.rejected = d["saved"], set(d["rejected"])
        return s

    def erase_all(self):
        """Forget everything learned about this user."""
        self.wardrobe.clear()
        self.saved.clear()
        self.rejected.clear()
        self.prefs = PreferenceModel()
        self.recommender = Recommender(self.prefs)
        self.profile = StyleProfile(language=self.profile.language)
