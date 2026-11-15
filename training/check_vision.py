"""Sanity check: auto-tagging accuracy of aura.vision on held-out Polyvore product images."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.encoder import load_encoder  # noqa: E402
from aura.taxonomy import slot_for_category_name  # noqa: E402
from aura.vision import Vision  # noqa: E402

v = Vision(load_encoder())
names = {int(l.split(" ", 1)[0]): l.strip().split(" ", 1)[1]
         for l in open(ROOT / "data/polyvore/category_id.txt", encoding="utf8")}
truth = {}
for o in json.load(open(ROOT / "data/polyvore/test_no_dup.json"))[:80]:
    for it in o["items"]:
        truth[f"{o['set_id']}_{it['index']}"] = (names[it["categoryid"]], it["name"])
fs = sorted((ROOT / "data/demo_raw").glob("*.jpg"))[: int(sys.argv[1]) if len(sys.argv) > 1 else 60]
ok = n = 0
t0 = time.time()
for f in fs:
    a = v.analyse(f.read_bytes())
    cat, name = truth[f.stem]
    ts = slot_for_category_name(cat)
    if ts != "other":
        n += 1
        ok += ts == a["slot"]
    if ts != a["slot"]:
        print(f"{cat[:24]:24} true={ts:9} pred={a['slot']:9} {a['name']:22} {a['style']:12} | {name[:40]}")
print(f"slot accuracy {ok}/{n}; {(time.time()-t0)/len(fs):.2f} s/img")
