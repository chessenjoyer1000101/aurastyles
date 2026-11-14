"""Run with:  python -m aura_stylist.demo

Uses a hand-entered wardrobe, so it needs no photos and no vision model.
"""
from . import AuraStylist, Context, Occasion, Reason, StyleProfile, build_item

WARDROBE = [
    ("t1", "t-shirt", "white"), ("t2", "dress shirt", "light blue"), ("t3", "sweater", "burgundy"),
    ("t4", "hoodie", "gray"), ("t5", "blouse", "cream"),
    ("b1", "jeans", "blue"), ("b2", "dress trousers", "navy"), ("b3", "chinos", "beige"),
    ("o1", "blazer", "navy"), ("o2", "coat", "gray"), ("o3", "denim jacket", "blue"),
    ("s1", "sneakers", "white"), ("s2", "loafers", "brown"), ("s3", "sandals", "brown"),
    ("g1", "tote bag", "black"), ("a1", "watch", "black"),
]


def show(title, rec):
    print(f"\n== {title}")
    for n, o in enumerate(rec.outfits, 1):
        print(f" {n}. [{o.score:.2f}] {o.explanation}")
    for note in rec.notes:
        print(f"    note: {note}")


def main():
    s = AuraStylist(StyleProfile(preferred_styles=["classic"], liked_colors=["navy"],
                                 avoided_colors=["orange"],
                                 occasions=[Occasion.OFFICE, Occasion.CASUAL, Occasion.FORMAL]))
    for i, cat, col in WARDROBE:
        s.add_item(build_item(i, cat, [col]))

    office = Context(Occasion.OFFICE, temperature_c=9, condition="rain")
    rec = s.recommend(office)
    show("Office, 9°C and rain", rec)
    show("Casual, 31°C", s.recommend(Context(Occasion.CASUAL, temperature_c=31)))

    s.dislike(rec.outfits[0], office, Reason.TOO_FORMAL)
    show("Office again, after 'too formal'", s.recommend(office))
    show("Challenge: restyle the hoodie", s.challenge(category="hoodie"))

    print("\n== Wardrobe gaps")
    for g in s.gaps():
        print(f" [{g.priority}] {g.message}")
    print("\n== Style directions")
    for st in s.style_identity():
        print(f" {st.style} ({st.share:.0%}{', tentative' if st.tentative else ''}): {st.evidence}")

    s.profile.language = "ar"
    show("بالعربية: عشاء", s.recommend(Context(Occasion.DINNER, temperature_c=20)))


if __name__ == "__main__":
    main()
