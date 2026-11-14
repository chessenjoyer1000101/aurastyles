"""Wardrobe gap analysis (US-011) and style identity discovery (US-008)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .colors import NEUTRALS
from .knowledge import (OCCASION_AR, OCCASION_EN, SLOT_AR, STYLE_AR, STYLES,
                        target_formality)
from .schema import Occasion, Slot, StyleProfile

BAND = 0.30   # how far an item's formality may sit from an occasion's target


@dataclass
class Gap:
    priority: str        # "essential" or "optional"
    slot: Slot
    occasion: object     # Occasion or None
    message: str


@dataclass
class StyleSuggestion:
    style: str
    share: float         # 0..1 share of the evidence
    evidence: str
    tentative: bool      # True => too little data to treat this as settled


def _t(lang, en, ar):
    return ar if lang == "ar" else en


def find_gaps(wardrobe: list, profile: StyleProfile) -> list:
    """Missing pieces, judged against the occasions the user actually dresses for.

    essential: an occasion for which no complete outfit can be built.
    optional:  an outfit is possible but variety is thin.
    Suggestions name a kind of garment, never a product.
    """
    lang = profile.language
    items = [i for i in wardrobe if i.slot is not None]
    occasions = [Occasion(o) for o in profile.occasions] or [Occasion.CASUAL]
    gaps, seen = [], set()

    def add(priority, slot, occ, en, ar):
        if (slot, occ) not in seen:
            seen.add((slot, occ))
            gaps.append(Gap(priority, slot, occ, _t(lang, en, ar)))

    for occ in occasions:
        target = target_formality(occ)
        fit = [i for i in items if abs(i.formality - target) <= BAND]
        n = Counter(i.slot for i in fit)
        en_o, ar_o = OCCASION_EN[occ], OCCASION_AR[occ]
        if n[Slot.ONE_PIECE] == 0:
            for slot, word in ((Slot.TOP, "top"), (Slot.BOTTOM, "bottom")):
                if n[slot] == 0:
                    add("essential", slot, occ,
                        f"Nothing in the wardrobe works as a {word} for {en_o}, "
                        f"so no complete outfit can be built for it.",
                        f"لا توجد {SLOT_AR[slot]} مناسبة {ar_o}، "
                        f"لذلك لا يمكن تكوين إطلالة كاملة لها.")
        if n[Slot.SHOES] == 0:
            add("essential", Slot.SHOES, occ,
                f"No shoes at the right level of dress for {en_o}.",
                f"لا يوجد حذاء بمستوى مناسب {ar_o}.")
        elif n[Slot.SHOES] == 1:
            add("optional", Slot.SHOES, occ,
                f"Only one pair of shoes suits {en_o}; a second pair would add variety.",
                f"يوجد حذاء واحد فقط مناسب {ar_o}؛ زوج ثانٍ يضيف تنوعاً.")

    bottoms = [i for i in items if i.slot is Slot.BOTTOM]
    tops = [i for i in items if i.slot is Slot.TOP]
    chroma_tops = sum(1 for i in tops if i.main_color and i.main_color.name not in NEUTRALS)
    if bottoms and chroma_tops >= 3 and not any(
            i.main_color and i.main_color.name in NEUTRALS for i in bottoms):
        add("optional", Slot.BOTTOM, None,
            "A bottom in a neutral colour would pair with most of your coloured tops.",
            "قطعة سفلية بلون محايد ستتناسق مع معظم القطع العلوية الملونة لديك.")
    if items and not any(i.slot is Slot.OUTER and i.warmth >= 0.6 for i in items):
        add("optional", Slot.OUTER, None,
            "There is no warm outer layer for cold days.",
            "لا توجد طبقة خارجية دافئة للأيام الباردة.")

    gaps.sort(key=lambda g: g.priority != "essential")
    return gaps


def discover_style(wardrobe: list, profile: StyleProfile, liked_outfits: list = (),
                   style_affinity: dict = None, top: int = 3, min_items: int = 8) -> list:
    """Suggested style directions with the evidence behind each.

    Evidence: what the user said they like (strongest), outfits they saved,
    then what they own. Results are marked tentative until the wardrobe and
    feedback are large enough, and are suggestions the user can reject or
    edit, never a fixed label.
    """
    lang = profile.language
    votes, why = Counter(), {s: Counter() for s in STYLES}
    items = [i for i in wardrobe if i.slot is not None]
    for i in items:
        for s in i.styles:
            votes[s] += 1
            why[s]["owned"] += 1
    for o in liked_outfits:
        for i in o.items:
            for s in i.styles:
                votes[s] += 2
                why[s]["liked"] += 1
    for s in profile.preferred_styles:
        votes[s] += 3
        why.setdefault(s, Counter())["stated"] += 1
    plain = [i for i in items if i.pattern == "solid" and i.main_color
             and i.main_color.name in NEUTRALS]
    if items and len(plain) / len(items) >= 0.6:
        votes["minimalist"] += len(plain) * 0.5
        why["minimalist"]["neutral"] += len(plain)
    for s, a in (style_affinity or {}).items():
        votes[s] = max(0.0, votes[s] * (1 + 2 * a))

    total = sum(votes.values())
    if not total:
        return []
    tentative = len(items) + 2 * len(liked_outfits) < min_items
    out = []
    for s, v in sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[:top]:
        if v <= 0:
            continue
        w, bits = why[s], []
        if w["stated"]:
            bits.append(_t(lang, "you named it as a preference", "ذكرته ضمن تفضيلاتك"))
        if w["liked"]:
            bits.append(_t(lang, f"{w['liked']} pieces in outfits you saved",
                           f"{w['liked']} قطعة في إطلالات حفظتها"))
        if w["owned"]:
            bits.append(_t(lang, f"{w['owned']} pieces in your wardrobe",
                           f"{w['owned']} قطعة في خزانتك"))
        if w["neutral"]:
            bits.append(_t(lang, "mostly plain, neutral colours", "غالبية الألوان محايدة وسادة"))
        name = STYLE_AR.get(s, s) if lang == "ar" else s
        out.append(StyleSuggestion(name, round(v / total, 3),
                                   ("، " if lang == "ar" else "; ").join(bits), tentative))
    return out
