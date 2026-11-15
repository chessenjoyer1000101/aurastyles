"""Outfit slots shared by training (Polyvore categories) and the app (wardrobe items)."""
from __future__ import annotations

import re

SLOTS = ["top", "bottom", "dress", "outerwear", "shoes", "bag", "jewelry",
         "hat", "eyewear", "scarf", "belt", "watch", "accessory", "other"]
SLOT_ID = {s: i for i, s in enumerate(SLOTS)}
OTHER = SLOT_ID["other"]

SLOT_LABEL = {
    "top": "Top", "bottom": "Bottom", "dress": "Dress / one-piece", "outerwear": "Outerwear / layer",
    "shoes": "Shoes", "bag": "Bag", "jewelry": "Jewelry", "hat": "Hat", "eyewear": "Eyewear",
    "scarf": "Scarf", "belt": "Belt", "watch": "Watch", "accessory": "Other accessory",
}

# Polyvore lists items top -> bottom -> shoes -> accessories; we keep the same order for display.
DISPLAY_ORDER = ["outerwear", "top", "dress", "bottom", "shoes", "bag", "hat", "eyewear",
                 "scarf", "belt", "watch", "jewelry", "accessory", "other"]

# Ordered (first match wins) rules mapping a Polyvore category name to a slot.
_RULES = [
    (r"bikini|swim|cover-up|bras?$|panties|intimates|shapewear|pajamas|robes|chemises|underwear|costume|maternity", "other"),
    (r"tech accessories|bags & cases|key rings|napkin|money clip|wallets?$|luggage|umbrella", "other"),
    (r"dress shirts|casual shirts|polos", "top"),
    (r"dress pants|casual pants", "bottom"),
    (r"dress shoes", "shoes"),
    (r"\b(jackets?|coats?|blazers|vests|outerwear|cardigans)\b", "outerwear"),
    (r"\b(dresses|gowns|rompers|jumpsuits|suits)\b", "dress"),
    (r"\b(tops|sweaters|t-shirts|blouses|sweatshirts|hoodies|tunics|shirts|camisoles)\b", "top"),
    (r"\b(jeans|shorts|skirts|pants|leggings)\b", "bottom"),
    (r"\b(booties|pumps|sandals|sneakers|flats|shoes|boots|moccasins|oxfords|slippers|flip flops|clogs)\b", "shoes"),
    (r"\b(bags|clutches|backpacks|handbags|briefcases)\b", "bag"),
    (r"\b(earrings|necklaces|bracelets|bangles|rings|jewelry|brooches|charms|cuff links)\b", "jewelry"),
    (r"\bwatches\b", "watch"),
    (r"\bhats\b", "hat"),
    (r"sunglasses|eyeglasses|eyewear", "eyewear"),
    (r"\bscarves\b", "scarf"),
    (r"\bbelts\b|suspenders", "belt"),
    (r"gloves|socks|tights|hosiery|hair accessories|\bties\b|bow ties|^accessories$|^men's accessories$", "accessory"),
]
_RULES = [(re.compile(p), s) for p, s in _RULES]


def slot_for_category_name(name: str) -> str:
    n = name.lower().strip()
    # "Men's Watches & Jewelry" etc.: watches rule sits after jewelry, handle explicitly
    if "watch" in n:
        return "watch"
    for rx, slot in _RULES:
        if rx.search(n):
            return slot
    return "other"


def load_category_slots(path) -> dict[int, int]:
    out = {}
    for line in open(path, encoding="utf8"):
        cid, name = line.strip().split(" ", 1)
        out[int(cid)] = SLOT_ID[slot_for_category_name(name)]
    return out
