"""Optional step 1b - text-embed the titles of training items that have no image in the mirror.

About 45 % of the *training* outfits have no images in `Marqo/polyvore`, but every
item still has its title in the json.  FashionSigLIP puts text and images in one
space, so train_compat.py (--pseudo) can map these title embeddings into the image
space with a ridge regression fitted on the ~94k real (title, image) pairs and use
them as extra, noisier training outfits.

Output: data/emb/text_missing.npz (ids, txt float16)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.encoder import load_encoder  # noqa: E402

have = set()
for f in (ROOT / "data/emb").glob("shard_*.npz"):
    have |= set(np.load(f)["ids"].tolist())
ids, names = [], []
for o in json.load(open(ROOT / "data/polyvore/train_no_dup.json")):
    for it in o["items"]:
        iid = f"{o['set_id']}_{it['index']}"
        if iid not in have:
            ids.append(iid)
            names.append(it["name"] or "")
print(len(ids), "train items without an image")
enc = load_encoder()
out = []
for i in range(0, len(names), 256):
    out.append(enc.embed_text(names[i:i + 256]).cpu().numpy().astype(np.float16))
np.savez(ROOT / "data/emb/text_missing.npz", ids=np.array(ids), txt=np.concatenate(out))
print("saved")
