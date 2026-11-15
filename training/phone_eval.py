"""Measure phone-photo handling on real phone photos, and train the slot classifier.

    python training/phone_eval.py raw silueta ...        # compare preprocessing variants
    python training/phone_eval.py silueta --save         # also write models/slot_head.pt

For every variant (see phone_embed.py), on the dataset's held-out *test* photos:
  zero-shot   slot accuracy of the app's original text-prompt tagging
  transfer    accuracy of a classifier trained ONLY on Polyvore product shots
              (how product-like the phone embeddings are: the domain-gap metric)
  +shift      same, after moving phone embeddings by (product mean - phone mean)
  head        accuracy of a classifier trained on product shots + phone train photos
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.taxonomy import OTHER, SLOT_ID, SLOTS, slot_for_category_name  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PHONE2SLOT = {"dress": "dress", "hat": "hat", "longsleeve": "top", "outwear": "outerwear", "pants": "bottom",
              "shirt": "top", "shoes": "shoes", "shorts": "bottom", "skirt": "bottom", "t-shirt": "top"}


def product_data():
    X, y = [], []
    for f in sorted((ROOT / "data/emb").glob("shard_*.npz")):
        z = np.load(f)
        s = np.array([SLOT_ID[slot_for_category_name(c)] for c in z["category"]])
        keep = s != OTHER
        X.append(z["img"][keep])
        y.append(s[keep])
    return torch.from_numpy(np.concatenate(X)).float().to(DEV), torch.from_numpy(np.concatenate(y)).to(DEV)


def phone_data(variant):
    z = np.load(ROOT / "data/phone" / f"emb_{variant}.npz")
    X = torch.from_numpy(z["emb"]).float().to(DEV)
    y = torch.tensor([SLOT_ID[PHONE2SLOT[l]] for l in z["label"]], device=DEV)
    return X, y, z["split"], float(z["mask_ok"].mean())


def fit(X, y, w=None, epochs=300, wd=1e-4):
    """multinomial logistic regression on L2-normalised embeddings (full batch)"""
    head = torch.nn.Linear(X.shape[1], len(SLOTS) - 1).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=wd)
    for _ in range(epochs):
        loss = F.cross_entropy(head(X) * 20, y, weight=None, reduction="none")
        loss = (loss * w).sum() / w.sum() if w is not None else loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


def acc(head, X, y):
    with torch.no_grad():
        return float((head(X).argmax(1) == y).float().mean())


def zero_shot(X, y):
    from aura.encoder import load_encoder
    from aura.vision import TYPE_NAMES, TYPES, Vision
    v = Vision(load_encoder())
    t2s = torch.tensor([SLOT_ID[TYPES[t]] for t in TYPE_NAMES], device=DEV)
    return float((t2s[(X @ v.type_emb.T.float()).argmax(1)] == y).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--save", action="store_true", help="save the head trained on the first variant")
    a = ap.parse_args()
    torch.manual_seed(0)
    PX, Py = product_data()
    perm = torch.randperm(len(PX), device=DEV)
    hold, tr = perm[:5000], perm[5000:]
    prod_head = fit(PX[tr], Py[tr])
    mu_prod = PX[tr].mean(0)
    print(f"product shots: {len(PX)} items; product-only classifier on held-out product shots: {acc(prod_head, PX[hold], Py[hold]):.3f}")
    print(f"{'variant':22} {'cutout ok':>9} {'zero-shot':>9} {'transfer':>9} {'+shift':>7} {'head':>6} {'head/prod':>9}")
    for k, variant in enumerate(a.variants):
        X, y, split, ok = phone_data(variant)
        trn = torch.from_numpy(split != "test").to(DEV)
        tst = ~trn
        shift = mu_prod - X[trn].mean(0)
        Xs = F.normalize(X + shift, dim=-1)
        # joint head: phone photos are few, so weight them up to ~1/3 of the loss
        XX, yy = torch.cat([PX[tr], X[trn]]), torch.cat([Py[tr], y[trn]])
        w = torch.cat([torch.ones(len(tr), device=DEV), torch.full((int(trn.sum()),), len(tr) / (2 * int(trn.sum())), device=DEV)])
        head = fit(XX, yy, w)
        print(f"{variant:22} {ok:9.1%} {zero_shot(X[tst], y[tst]):9.3f} {acc(prod_head, X[tst], y[tst]):9.3f} "
              f"{acc(prod_head, Xs[tst], y[tst]):7.3f} {acc(head, X[tst], y[tst]):6.3f} {acc(head, PX[hold], Py[hold]):9.3f}")
        if a.save and k == 0:
            torch.save({"state": head.state_dict(), "slots": SLOTS[:-1], "variant": variant,
                        "phone_test_acc": acc(head, X[tst], y[tst]), "product_acc": acc(head, PX[hold], Py[hold])},
                       ROOT / "models" / "slot_head.pt")
            print("saved models/slot_head.pt")


if __name__ == "__main__":
    main()
