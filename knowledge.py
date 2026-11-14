"""Garment knowledge base: what each category is, and what each occasion asks for.

Values are priors, not rules. The feedback model shifts them per user.
"""
from __future__ import annotations

from .schema import ColorInfo, Occasion, Slot, WardrobeItem

T, B, O, X, S, G, A = (Slot.TOP, Slot.BOTTOM, Slot.ONE_PIECE, Slot.OUTER,
                       Slot.SHOES, Slot.BAG, Slot.ACCESSORY)

# category: (slot, formality, warmth, coverage, styles, arabic name)
_C = {
    # tops
    "t-shirt":        (T, .20, .20, .40, ["casual", "minimalist", "sporty"], "تيشيرت"),
    "polo shirt":     (T, .45, .25, .40, ["casual", "classic"], "قميص بولو"),
    "dress shirt":    (T, .80, .30, .75, ["classic", "elegant"], "قميص رسمي"),
    "blouse":         (T, .70, .30, .65, ["elegant", "classic"], "بلوزة"),
    "tank top":       (T, .10, .05, .15, ["casual", "sporty"], "توب بدون أكمام"),
    "sweater":        (T, .50, .75, .80, ["classic", "casual", "minimalist"], "كنزة"),
    "turtleneck":     (T, .60, .80, .90, ["minimalist", "classic"], "كنزة بياقة عالية"),
    "hoodie":         (T, .10, .75, .80, ["casual", "sporty", "contemporary"], "هودي"),
    "tunic":          (T, .50, .35, .90, ["elegant", "casual"], "تونيك"),
    # bottoms
    "jeans":          (B, .30, .50, .90, ["casual", "contemporary"], "جينز"),
    "dress trousers": (B, .80, .45, .90, ["classic", "elegant"], "بنطلون رسمي"),
    "chinos":         (B, .55, .40, .90, ["classic", "casual"], "بنطلون تشينو"),
    "shorts":         (B, .10, .05, .30, ["casual", "sporty"], "شورت"),
    "leggings":       (B, .10, .40, .60, ["sporty", "casual"], "ليقنز"),
    "sweatpants":     (B, .05, .60, .90, ["sporty", "casual"], "بنطلون رياضي"),
    "mini skirt":     (B, .40, .15, .30, ["contemporary", "casual"], "تنورة قصيرة"),
    "midi skirt":     (B, .60, .30, .70, ["elegant", "classic"], "تنورة متوسطة"),
    "maxi skirt":     (B, .60, .40, 1.0, ["elegant", "classic"], "تنورة طويلة"),
    # one-piece
    "casual dress":   (O, .40, .25, .55, ["casual", "contemporary"], "فستان يومي"),
    "maxi dress":     (O, .60, .35, .95, ["elegant", "casual"], "فستان طويل"),
    "evening dress":  (O, .95, .30, .70, ["elegant"], "فستان سهرة"),
    "jumpsuit":       (O, .55, .35, .80, ["contemporary", "elegant"], "جمبسوت"),
    "abaya":          (O, .65, .45, 1.0, ["classic", "elegant"], "عباية"),
    "thobe":          (O, .70, .40, 1.0, ["classic"], "ثوب"),
    # outerwear
    "blazer":         (X, .85, .50, .80, ["classic", "elegant"], "بليزر"),
    "cardigan":       (X, .50, .55, .75, ["classic", "casual"], "كارديغان"),
    "denim jacket":   (X, .30, .50, .80, ["casual", "contemporary"], "جاكيت جينز"),
    "leather jacket": (X, .40, .60, .80, ["contemporary"], "جاكيت جلد"),
    "trench coat":    (X, .70, .65, .95, ["classic", "elegant"], "معطف ترنش"),
    "coat":           (X, .70, .95, .95, ["classic", "elegant"], "معطف"),
    "puffer jacket":  (X, .20, 1.0, .90, ["casual", "sporty"], "جاكيت منفوخ"),
    # shoes
    "sneakers":       (S, .20, .50, .50, ["casual", "sporty", "contemporary"], "حذاء رياضي"),
    "loafers":        (S, .70, .45, .50, ["classic"], "لوفر"),
    "dress shoes":    (S, .90, .50, .50, ["classic", "elegant"], "حذاء رسمي"),
    "heels":          (S, .85, .30, .50, ["elegant"], "كعب عالٍ"),
    "ballet flats":   (S, .60, .30, .50, ["classic", "elegant"], "حذاء باليرينا"),
    "sandals":        (S, .20, .05, .50, ["casual"], "صندل"),
    "boots":          (S, .50, .90, .50, ["classic", "contemporary"], "بوت"),
    # bags
    "handbag":        (G, .70, .50, .50, ["classic", "elegant"], "حقيبة يد"),
    "tote bag":       (G, .45, .50, .50, ["casual", "minimalist"], "حقيبة توت"),
    "backpack":       (G, .20, .50, .50, ["casual", "sporty"], "حقيبة ظهر"),
    "clutch":         (G, .90, .50, .50, ["elegant"], "حقيبة سهرة"),
    # accessories
    "scarf":          (A, .50, .70, .50, ["classic"], "وشاح"),
    "hijab":          (A, .50, .40, 1.0, ["classic", "elegant"], "حجاب"),
    "belt":           (A, .55, .50, .50, ["classic"], "حزام"),
    "watch":          (A, .60, .50, .50, ["classic", "minimalist"], "ساعة"),
    "necklace":       (A, .70, .50, .50, ["elegant"], "قلادة"),
    "tie":            (A, .95, .50, .50, ["classic"], "ربطة عنق"),
    "cap":            (A, .15, .40, .50, ["casual", "sporty"], "قبعة"),
    "sunglasses":     (A, .30, .50, .50, ["contemporary"], "نظارة شمسية"),
}

CATEGORIES = {k: dict(slot=v[0], formality=v[1], warmth=v[2], coverage=v[3],
                      styles=v[4], ar=v[5]) for k, v in _C.items()}
OPEN_SHOES = {"sandals"}

PATTERNS = ["solid", "striped", "checked", "floral", "polka dot", "graphic", "animal print"]
STYLES = ["classic", "minimalist", "casual", "elegant", "contemporary", "sporty"]

# occasion: target formality
OCCASION_FORMALITY = {
    Occasion.OFFICE: .75, Occasion.CASUAL: .25, Occasion.DINNER: .60,
    Occasion.FORMAL: .92, Occasion.TRAVEL: .25, Occasion.SPECIAL: .80,
}
DRESS_CODE_FORMALITY = {"casual": .25, "smart casual": .55, "business": .80, "formal": .92}

SLOT_AR = {Slot.TOP: "قطعة علوية", Slot.BOTTOM: "قطعة سفلية", Slot.ONE_PIECE: "قطعة كاملة",
           Slot.OUTER: "طبقة خارجية", Slot.SHOES: "حذاء", Slot.BAG: "حقيبة",
           Slot.ACCESSORY: "إكسسوار"}
OCCASION_AR = {Occasion.OFFICE: "للعمل", Occasion.CASUAL: "للخروج اليومي",
               Occasion.DINNER: "للعشاء", Occasion.FORMAL: "للمناسبات الرسمية",
               Occasion.TRAVEL: "للسفر", Occasion.SPECIAL: "للمناسبات الخاصة"}
OCCASION_EN = {Occasion.OFFICE: "the office", Occasion.CASUAL: "a casual day",
               Occasion.DINNER: "dinner", Occasion.FORMAL: "a formal event",
               Occasion.TRAVEL: "travel", Occasion.SPECIAL: "a special event"}
COLOR_AR = {"black": "أسود", "white": "أبيض", "gray": "رمادي", "navy": "كحلي",
            "beige": "بيج", "cream": "كريمي", "brown": "بني", "red": "أحمر",
            "burgundy": "عنابي", "pink": "وردي", "orange": "برتقالي", "yellow": "أصفر",
            "green": "أخضر", "olive": "زيتي", "teal": "أزرق مخضر", "blue": "أزرق",
            "light blue": "أزرق فاتح", "purple": "بنفسجي"}
STYLE_AR = {"classic": "كلاسيكي", "minimalist": "بسيط", "casual": "كاجوال",
            "elegant": "أنيق", "contemporary": "عصري", "sporty": "رياضي"}


def target_formality(occasion: Occasion, dress_code=None) -> float:
    if dress_code and dress_code.lower() in DRESS_CODE_FORMALITY:
        return DRESS_CODE_FORMALITY[dress_code.lower()]
    return OCCASION_FORMALITY[Occasion(occasion)]


def target_warmth(temperature_c: float) -> float:
    """How warm an outfit should be at a given temperature (piecewise linear)."""
    pts = [(3, .95), (10, .70), (18, .45), (24, .25), (30, .10)]
    if temperature_c <= pts[0][0]:
        return pts[0][1]
    for (t0, w0), (t1, w1) in zip(pts, pts[1:]):
        if temperature_c <= t1:
            return w0 + (w1 - w0) * (temperature_c - t0) / (t1 - t0)
    return pts[-1][1]


def build_item(item_id: str, category: str, colors=None, pattern: str = "solid",
               **overrides) -> WardrobeItem:
    """Create an item whose attributes are derived from its category.

    `colors` is a list of ColorInfo or colour names. Unknown categories give an
    item with no slot, flagged for review; the recommender ignores it until the
    user corrects it.
    """
    from .colors import COLOR_RGB
    cols = []
    for c in colors or []:
        cols.append(c if isinstance(c, ColorInfo) else ColorInfo(c, COLOR_RGB[c], 1.0))
    spec = CATEGORIES.get(category)
    if spec is None:
        item = WardrobeItem(id=item_id, category="unknown", slot=None, colors=cols,
                            pattern=pattern, needs_review=True, confidence=0.0)
    else:
        item = WardrobeItem(id=item_id, category=category, slot=spec["slot"], colors=cols,
                            pattern=pattern, formality=spec["formality"],
                            warmth=spec["warmth"], coverage=spec["coverage"],
                            styles=list(spec["styles"]), open_shoe=category in OPEN_SHOES)
    for k, v in overrides.items():
        setattr(item, k, v)
    return item
