"""Outfit generation, scoring, diversification and explanation (US-004..007, US-010).

Pipeline:
  1. filter the wardrobe (identified items only, modesty preference),
  2. enumerate core outfits: top + bottom + shoes, or one-piece + shoes,
  3. score every core with the preference model, keep the best,
  4. extend each with an outer layer, bag and accessory when that helps,
  5. pick k results that are good and different from each other (MMR),
  6. explain each result in the user's language.

Only items from the wardrobe are ever used. If a complete outfit cannot be
built, the result says which slot is missing instead of inventing one.
"""
from __future__ import annotations

from collections import Counter
from itertools import product

import numpy as np

from .colors import harmony_score
from .feedback import PreferenceModel
from .knowledge import (CATEGORIES, COLOR_AR, OCCASION_AR, OCCASION_EN, STYLE_AR,
                        target_formality, target_warmth)
from .schema import Context, Outfit, Recommendation, Slot, StyleProfile

BODY = (Slot.TOP, Slot.BOTTOM, Slot.ONE_PIECE)
MAX_CORES, KEEP_CORES, PER_SLOT_CAP = 20000, 150, 25


class Recommender:
    def __init__(self, prefs: PreferenceModel = None):
        self.prefs = prefs or PreferenceModel()

    # ------------------------------------------------------------------
    # features
    # ------------------------------------------------------------------
    def features(self, items: list, profile: StyleProfile, ctx: Context) -> dict:
        p = self.prefs
        body = [i for i in items if i.slot in BODY]
        outer = [i for i in items if i.slot is Slot.OUTER]

        # formality: close to the occasion's level, and consistent across pieces
        target = np.clip(target_formality(ctx.occasion, ctx.dress_code)
                         + p.formality_offset.get(ctx.occasion.value, 0.0), 0, 1)
        wts = [0.5 if i.slot in (Slot.BAG, Slot.ACCESSORY) else 1.0 for i in items]
        mean_f = float(np.average([i.formality for i in items], weights=wts))
        main = [i.formality for i in items if i.slot not in (Slot.BAG, Slot.ACCESSORY)]
        fit = 1 - min(1.0, abs(mean_f - target) / 0.45)
        consistency = 1 - min(1.0, (max(main) - min(main)) / 0.8) if main else 1.0
        formality = 0.75 * fit + 0.25 * consistency

        # weather: neutral when no weather data was supplied
        if ctx.temperature_c is None:
            weather = 0.5
        else:
            want = np.clip(target_warmth(ctx.temperature_c) + p.warmth_offset, 0, 1)
            have = float(np.mean([i.warmth for i in body])) if body else 0.5
            if outer:
                have = min(1.0, have + 0.35 * max(o.warmth for o in outer))
            weather = 1 - min(1.0, abs(have - want) * 1.6)
            wet = (ctx.condition or "").lower() in ("rain", "snow")
            if any(i.open_shoe for i in items) and (wet or ctx.temperature_c < 12):
                weather -= 0.4
            if wet and outer:
                weather += 0.1
            weather = float(np.clip(weather, 0, 1))

        # style: pieces share a direction, and it is one the user likes
        tags = Counter(s for i in items for s in i.styles)
        if tags:
            dom, n = max(sorted(tags.items()), key=lambda kv: kv[1])
            coherence = n / len(items)
            aff = p.style_affinity.get(dom, 0.0)
        else:
            coherence, aff = 0.5, 0.0
        if profile.preferred_styles:
            pref = np.mean([bool(set(i.styles) & set(profile.preferred_styles)) for i in items])
        else:
            pref = 0.5
        style = float(np.clip(0.5 * coherence + 0.5 * pref + aff, 0, 1))

        # colour preference
        votes = []
        for i in items:
            names = {c.name for c in i.colors[:2]}
            if names & set(profile.avoided_colors):
                votes.append(-1.0)
            elif names & set(profile.liked_colors):
                votes.append(1.0)
            else:
                votes.append(0.0)
        color_pref = 0.5 + 0.5 * float(np.mean(votes))

        bold = sum(1 for i in items if i.pattern != "solid")
        pattern = {0: 0.85, 1: 1.0, 2: 0.55}.get(bold, 0.25)
        novelty = 1 - float(np.mean([min(i.wear_count, 5) / 5 for i in items]))

        return {"harmony": harmony_score([i.main_color for i in items]),
                "formality": float(formality), "weather": weather, "style": style,
                "color_pref": color_pref, "pattern": pattern, "novelty": novelty}

    def _score(self, items, profile, ctx, challenge):
        f = self.features(items, profile, ctx)
        return self.prefs.score(f, items, challenge), f

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def recommend(self, wardrobe: list, profile: StyleProfile, ctx: Context, k: int = 3,
                  must_include: str = None, exclude: set = (), challenge: bool = False
                  ) -> Recommendation:
        notes = []
        lang = profile.language
        usable = [i for i in wardrobe if i.slot is not None]
        skipped = len(wardrobe) - len(usable)
        if skipped:
            notes.append(_t(lang, f"{skipped} item(s) are not identified yet and were left out.",
                            f"تم استبعاد {skipped} قطعة لم يتم التعرف عليها بعد."))
        if profile.min_coverage is not None:
            usable = [i for i in usable
                      if i.slot not in BODY or i.coverage >= profile.min_coverage]

        by = {s: [i for i in usable if i.slot is s] for s in Slot}
        if profile.avoided_colors:      # leave avoided colours out wherever a choice remains
            avoid = set(profile.avoided_colors)
            for s, pool in by.items():
                keep = [i for i in pool if not (i.main_color and i.main_color.name in avoid)
                        or i.id == must_include]
                if keep or s not in (*BODY, Slot.SHOES):
                    by[s] = keep
        forced = next((i for i in usable if i.id == must_include), None)
        if must_include and forced is None:
            notes.append(_t(lang, "The requested item is not available for this outfit.",
                            "القطعة المطلوبة غير متاحة لهذه الإطلالة."))
            return Recommendation([], notes)
        if forced is not None and forced.slot in (*BODY, Slot.SHOES):
            by[forced.slot] = [forced]
            if forced.slot is Slot.ONE_PIECE:
                by[Slot.TOP], by[Slot.BOTTOM] = [], []
            elif forced.slot in (Slot.TOP, Slot.BOTTOM):
                by[Slot.ONE_PIECE] = []

        shoes = by[Slot.SHOES] or [None]
        if shoes == [None]:
            notes.append(_t(lang, "No shoes in the wardrobe yet, so outfits are shown without them.",
                            "لا توجد أحذية في الخزانة بعد، لذا تُعرض الإطلالات بدونها."))

        def cap(pool):
            if len(pool) <= PER_SLOT_CAP:
                return pool
            return sorted(pool, key=lambda i: -self._score([i], profile, ctx, challenge)[0]
                          )[:PER_SLOT_CAP]

        tops, bottoms, ones = by[Slot.TOP], by[Slot.BOTTOM], by[Slot.ONE_PIECE]
        if (len(tops) * len(bottoms) + len(ones)) * len(shoes) > MAX_CORES:
            tops, bottoms, ones, shoes = cap(tops), cap(bottoms), cap(ones), cap(shoes)
        cores = [[t, b] for t, b in product(tops, bottoms)] + [[o] for o in ones]
        if not cores:
            missing = ("tops" if not tops else "bottoms") if not ones else ""
            notes.append(_t(lang,
                            f"A complete outfit needs a top and a bottom, or a one-piece. "
                            f"The wardrobe has no suitable {missing}.",
                            "الإطلالة الكاملة تحتاج قطعة علوية وسفلية أو قطعة كاملة، "
                            "ولا توجد قطع مناسبة كافية في الخزانة."))
            return Recommendation([], notes)

        scored = []
        for core, sh in product(cores, shoes):
            items = core + ([sh] if sh is not None else [])
            scored.append((self._score(items, profile, ctx, challenge)[0], items))
        scored.sort(key=lambda t: -t[0])

        outfits = []
        for _, items in scored[:KEEP_CORES]:
            items = self._extend(items, by, profile, ctx, challenge, forced)
            if forced is not None and forced not in items:
                continue
            s, f = self._score(items, profile, ctx, challenge)
            o = Outfit(items=items, score=float(s), features=f)
            if o.key not in exclude:
                outfits.append(o)

        chosen = sorted(self._diversify(outfits, k), key=lambda o: -o.score)
        if ctx.temperature_c is None:
            notes.append(_t(lang, "No weather data was available, so weather was not considered.",
                            "لم تتوفر بيانات الطقس، لذلك لم تؤخذ في الاعتبار."))
        for o in chosen:
            o.explanation = explain(o, profile, ctx, challenge)
        return Recommendation(chosen, notes)

    def _extend(self, items, by, profile, ctx, challenge, forced):
        """Add an outer layer, bag and accessory when doing so does not hurt the score."""
        best, _ = self._score(items, profile, ctx, challenge)
        for slot in (Slot.OUTER, Slot.BAG, Slot.ACCESSORY):
            if forced is not None and forced.slot is slot:
                items = items + [forced]
                best, _ = self._score(items, profile, ctx, challenge)
                continue
            pool = [i for i in by[slot] if i.category not in profile.always_include]
            # an outer layer must earn its place; bags and accessories only need not to hurt
            margin = 0.0 if slot is Slot.OUTER else 0.01
            pick, pick_s = None, best - margin
            for cand in pool:
                s, _ = self._score(items + [cand], profile, ctx, challenge)
                if s > pick_s:
                    pick, pick_s = cand, s
            if pick is not None:
                items, best = items + [pick], pick_s
        for cat in profile.always_include:             # e.g. hijab with every outfit
            pool = [i for i in by[Slot.ACCESSORY] + by[Slot.OUTER]
                    if i.category == cat and i not in items]
            if pool:
                items = items + [max(pool, key=lambda c: self._score(
                    items + [c], profile, ctx, challenge)[0])]
        return items

    @staticmethod
    def _diversify(outfits: list, k: int, lam: float = 0.35) -> list:
        """Maximal marginal relevance: high score, low overlap with earlier picks."""
        pool = sorted(outfits, key=lambda o: (-o.score, o.key))
        seen, uniq = set(), []
        for o in pool:
            if o.key not in seen:
                seen.add(o.key)
                uniq.append(o)
        chosen = []
        while uniq and len(chosen) < k:
            def value(o):
                ids = set(o.item_ids())
                overlap = max((len(ids & set(c.item_ids())) / len(ids | set(c.item_ids()))
                               for c in chosen), default=0.0)
                return o.score - lam * overlap
            pick = max(uniq, key=value)
            chosen.append(pick)
            uniq.remove(pick)
        return chosen


# ----------------------------------------------------------------------
# explanations
# ----------------------------------------------------------------------
def _t(lang, en, ar):
    return ar if lang == "ar" else en


def describe_item(item, lang="en") -> str:
    color = item.main_color.name if item.main_color else ""
    if lang == "ar":
        cat = CATEGORIES.get(item.category, {}).get("ar", item.category)
        return f"{cat} بلون {COLOR_AR.get(color, color)}" if color else cat
    return f"{color} {item.category}".strip()


def explain(outfit: Outfit, profile: StyleProfile, ctx: Context, challenge=False) -> str:
    """Plain-language reasons, built from the features that scored well (US-009)."""
    lang, f = profile.language, outfit.features
    sep = "، " if lang == "ar" else ", "
    parts = [_t(lang, "Wear: ", "الإطلالة: ")
             + sep.join(describe_item(i, lang) for i in outfit.items) + "."]

    names = list(dict.fromkeys(i.main_color.name for i in outfit.items if i.main_color))
    palette = sep.join(COLOR_AR.get(n, n) if lang == "ar" else n for n in names)
    if f["harmony"] >= 0.75 and palette:
        parts.append(_t(lang, f"The colours sit well together ({palette}).",
                        f"الألوان متناسقة ({palette})."))
    if f["formality"] >= 0.70:
        parts.append(_t(lang, f"The level of dress suits {OCCASION_EN[ctx.occasion]}.",
                        f"مستوى الأناقة مناسب {OCCASION_AR[ctx.occasion]}."))
    if ctx.temperature_c is not None and f["weather"] >= 0.65:
        parts.append(_t(lang, f"It is warm enough, and not too warm, for {ctx.temperature_c:g}°C.",
                        f"مناسبة لدرجة حرارة {ctx.temperature_c:g}°م."))
    liked = [s for s in profile.preferred_styles if any(s in i.styles for i in outfit.items)]
    if liked and f["style"] >= 0.6:
        s = liked[0]
        parts.append(_t(lang, f"It follows your {s} preference.",
                        f"تتماشى مع ذوقك ال{STYLE_AR.get(s, s)}."))
    if f["pattern"] == 1.0:
        parts.append(_t(lang, "One patterned piece gives it a focal point.",
                        "قطعة واحدة منقوشة تمنحها نقطة تركيز."))
    if challenge and f["novelty"] >= 0.7:
        parts.append(_t(lang, "It uses pieces you rarely wear.",
                        "تستخدم قطعاً قلّما ترتديها."))
    return " ".join(parts)
