"""Step 3 - build the app's sample wardrobe from held-out Polyvore *test* images (data/demo_raw).

Runs every image through the same pipeline as a real upload and keeps a balanced
selection, so "Try a sample wardrobe" behaves exactly like a user's own closet.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.encoder import load_encoder  # noqa: E402
from aura.taxonomy import slot_for_category_name  # noqa: E402
from aura.vision import Vision  # noqa: E402

WANT = {"top": 9, "bottom": 7, "dress": 3, "outerwear": 5, "shoes": 7, "bag": 4, "jewelry": 3,
        "hat": 2, "eyewear": 2, "scarf": 1, "belt": 1, "watch": 1}
OUT = ROOT / "app" / "demo"
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob("*"):
    f.unlink()

names = {int(l.split(" ", 1)[0]): l.strip().split(" ", 1)[1]
         for l in open(ROOT / "data/polyvore/category_id.txt", encoding="utf8")}
truth = {}
for o in json.load(open(ROOT / "data/polyvore/test_no_dup.json"))[:80]:
    for it in o["items"]:
        truth[f"{o['set_id']}_{it['index']}"] = slot_for_category_name(names[it["categoryid"]])

v = Vision(load_encoder())
files = sorted((ROOT / "data/demo_raw").glob("*.jpg"))
random.Random(7).shuffle(files)
have = {s: 0 for s in WANT}
meta = []
for f in files:
    slot = truth.get(f.stem)
    if slot not in WANT or have[slot] >= WANT[slot]:
        continue
    a = v.analyse(f.read_bytes())
    if a["slot"] != slot:  # keep only items the auto-tagger agrees on -> a clean demo
        continue
    have[slot] += 1
    iid = f"d{len(meta):03d}"
    (OUT / f"{iid}.png").write_bytes(a["png"])
    np.save(OUT / f"{iid}.npy", a["emb"])
    meta.append(dict(id=iid, name=a["name"], type=a["type"], slot=a["slot"], style=a["style"],
                     color=a["color"], colors=a["colors"]))
(OUT / "items.json").write_text(json.dumps(meta), "utf8")
print(len(meta), "demo items", have)
