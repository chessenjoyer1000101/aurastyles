"""Step 2 - train the outfit compatibility transformer on Polyvore.

Two objectives share one scoring head:
  * compatibility prediction (BCE): real outfits vs. corrupted ones
  * fill-in-the-blank ranking (4-way cross-entropy): the real item must out-score
    three substitutes placed in the same position

Negatives are mostly *slot-preserving* (a top is swapped for another top), so the
model has to learn style/colour compatibility rather than "does this outfit have
two pairs of shoes" - the app's outfit templates already guarantee structure.

Evaluated on the official test files from Han et al. 2017:
  fill_in_blank_test.json (paper Bi-LSTM+VSE: 68.6 %) and
  fashion_compatibility_prediction.txt (paper AUC: 0.90).
"""
import argparse
import json
import random  # noqa: F401
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.model import OutfitCompat, save  # noqa: E402
from aura.taxonomy import OTHER, SLOTS, load_category_slots  # noqa: E402

DATA = ROOT / "data"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXN = 8


def load_embeddings(pseudo=False):
    ids, embs, txts = [], [], []
    for f in sorted((DATA / "emb").glob("shard_*.npz")):
        z = np.load(f)
        ids += z["ids"].tolist()
        embs.append(z["img"])
        txts.append(z["txt"])
    emb = torch.from_numpy(np.concatenate(embs)).float()
    n_real = len(emb)
    if pseudo:
        # ridge regression title-embedding -> image-embedding, fitted on real pairs,
        # then applied to the titles of training items that have no image
        X = torch.from_numpy(np.concatenate(txts)).float().to(DEV)
        Y = emb.to(DEV)
        mx, my = X.mean(0), Y.mean(0)
        A = (X - mx).T @ (X - mx) + 1.0 * torch.eye(X.shape[1], device=DEV)
        W = torch.linalg.solve(A, (X - mx).T @ (Y - my))
        z = np.load(DATA / "emb" / "text_missing.npz")
        P = F.normalize((torch.from_numpy(z["txt"]).float().to(DEV) - mx) @ W + my, dim=-1).cpu()
        fit = F.cosine_similarity((X - mx) @ W + my, Y).mean().item()
        print(f"pseudo image embeddings for {len(P)} title-only items (fit cos {fit:.3f})")
        ids += z["ids"].tolist()
        emb = torch.cat([emb, P])
    return {s: i for i, s in enumerate(ids)}, emb, n_real


def load_outfits(split, row, cat2slot):
    """-> list of (rows LongTensor, slots LongTensor), dict item_id -> slot"""
    outfits, slot_of = [], {}
    missing = total = 0
    for o in json.load(open(DATA / "polyvore" / f"{split}_no_dup.json")):
        r, s = [], []
        for it in o["items"]:
            iid = f"{o['set_id']}_{it['index']}"
            total += 1
            sl = cat2slot.get(it["categoryid"], OTHER)
            slot_of[iid] = sl
            if iid not in row:
                missing += 1
                continue
            r.append(row[iid])
            s.append(sl)
        if len(r) >= 2:
            outfits.append((torch.tensor(r), torch.tensor(s)))
    print(f"{split}: {len(outfits)} outfits, {total} items, {missing} without embedding")
    return outfits, slot_of


class Sampler:
    """Vectorised (GPU) sampler.  Every method returns (rows [B,8], slots [B,8], pad [B,8])."""

    def __init__(self, outfits, lookup):
        N = len(outfits)
        self.rows = torch.zeros(N, MAXN, dtype=torch.long)
        self.slots = torch.zeros(N, MAXN, dtype=torch.long)
        self.len = torch.zeros(N, dtype=torch.long)
        for k, (r, s) in enumerate(outfits):
            n = min(len(r), MAXN)
            self.rows[k, :n], self.slots[k, :n], self.len[k] = r[:n], s[:n], n
        allr = torch.cat([o[0] for o in outfits])
        alls = torch.cat([o[1] for o in outfits])
        order = torch.argsort(alls, stable=True)
        self.pool = allr[order].to(DEV)
        cnt = torch.bincount(alls, minlength=len(SLOTS))
        self.cnt = cnt.to(DEV)
        self.off = (torch.cumsum(cnt, 0) - cnt).to(DEV)
        self.rows, self.slots, self.len = self.rows.to(DEV), self.slots.to(DEV), self.len.to(DEV)
        self.lookup = lookup.to(DEV)
        self.ar = torch.arange(MAXN, device=DEV)

    def rand_slot(self, slots):
        c = self.cnt[slots].clamp(min=1)
        return self.pool[(self.off[slots] + (torch.rand(slots.shape, device=DEV) * c).long()).clamp(max=len(self.pool) - 1)]

    def rand_any(self, shape):
        return self.pool[torch.randint(len(self.pool), shape, device=DEV)]

    def positive(self, B):
        o = torch.randint(len(self.rows), (B,), device=DEV)
        rows, slots = self.rows[o], self.slots[o]
        valid = self.ar[None] < self.len[o][:, None]
        # wearable-only view (what a real wardrobe looks like)
        w = valid & (slots != OTHER)
        use = (torch.rand(B, device=DEV) < 0.5) & (w.sum(1) >= 2)
        valid = torch.where(use[:, None], w, valid)
        # partial outfit: keep a random subset of k >= 2 items
        n = valid.sum(1)
        k = 2 + (torch.rand(B, device=DEV) * (n - 1)).long().clamp(max=(n - 2).clamp(min=0))
        k = torch.where(torch.rand(B, device=DEV) < 0.5, k, n)
        rank = torch.rand(B, MAXN, device=DEV).masked_fill(~valid, 2.0).argsort(1).argsort(1)
        valid = valid & (rank < k[:, None])
        return rows, slots, ~valid

    def negative(self, B):
        rows, slots, pad = self.positive(B)
        valid = ~pad
        n = valid.sum(1)
        u = torch.rand(B, device=DEV)
        rank = torch.rand(B, MAXN, device=DEV).masked_fill(pad, 2.0).argsort(1).argsort(1)
        k = 1 + (torch.rand(B, device=DEV) * (n // 2).clamp(min=1)).long().clamp(max=((n // 2).clamp(min=1) - 1))
        swap = valid & ((u < 0.4)[:, None] | (rank < k[:, None]))
        rows = torch.where(swap, self.rand_slot(slots), rows)
        # unstructured random sets (the paper's test negatives)
        rnd = u >= 0.8
        rr = self.rand_any((B, MAXN))
        rn = 3 + (torch.rand(B, device=DEV) * (MAXN - 2)).long().clamp(max=MAXN - 3)
        rows = torch.where(rnd[:, None], rr, rows)
        slots = torch.where(rnd[:, None], self.lookup[rr], slots)
        pad = torch.where(rnd[:, None], self.ar[None] >= rn[:, None], pad)
        return rows, slots, pad

    def fitb(self, B):
        """-> [B*4, 8] with the real outfit first in every group of four"""
        rows, slots, pad = self.positive(B)
        pos = torch.rand(B, MAXN, device=DEV).masked_fill(pad, -1.0).argmax(1)
        R, S, P = [rows], [slots], [pad]
        bi = torch.arange(B, device=DEV)
        for _ in range(3):
            r2, s2 = rows.clone(), slots.clone()
            same = torch.rand(B, device=DEV) < 0.5
            cand = torch.where(same, self.rand_slot(slots[bi, pos]), self.rand_any((B,)))
            r2[bi, pos] = cand
            s2[bi, pos] = torch.where(same, slots[bi, pos], self.lookup[cand])
            R.append(r2); S.append(s2); P.append(pad)
        st = lambda L: torch.stack(L, 1).reshape(B * 4, MAXN)
        return st(R), st(S), st(P)


def collate(samples, emb):
    B = len(samples)
    rows = torch.zeros(B, MAXN, dtype=torch.long)
    slots = torch.zeros(B, MAXN, dtype=torch.long)
    pad = torch.ones(B, MAXN, dtype=torch.bool)
    for b, (r, s) in enumerate(samples):
        n = min(len(r), MAXN)
        rows[b, :n], slots[b, :n], pad[b, :n] = r[:n], s[:n], False
    rows = rows.to(DEV)
    return emb[rows], slots.to(DEV), pad.to(DEV)


@torch.no_grad()
def score(model, samples, emb, bs=1024):
    out = []
    for i in range(0, len(samples), bs):
        out.append(model(*collate(samples[i:i + bs], emb)))
    return torch.cat(out).float().cpu().numpy()


def official_eval(model, emb, row, slot_of):
    def mk(ids):
        ids = [i for i in ids if i in row]
        return torch.tensor([row[i] for i in ids]), torch.tensor([slot_of.get(i, OTHER) for i in ids])

    # The image mirror does not contain every Polyvore item.  A FITB question is only
    # used when all four candidate answers (and at least one context item) have an
    # image; a compatibility sample when at least two of its items do.
    qs = json.load(open(DATA / "polyvore" / "fill_in_blank_test.json"))
    samples = []
    for q in qs:
        if not all(a in row for a in q["answers"]) or not any(i in row for i in q["question"]):
            continue
        for a in q["answers"]:
            samples.append(mk(list(q["question"]) + [a]))
    sc = score(model, samples, emb).reshape(-1, 4)
    fitb = float((sc.argmax(1) == 0).mean())

    labels, samples = [], []
    for line in open(DATA / "polyvore" / "fashion_compatibility_prediction.txt"):
        p = line.split()
        r, s = mk(p[1:])
        if len(r) >= 2:
            labels.append(int(p[0]))
            samples.append((r, s))
    auc = float(roc_auc_score(labels, score(model, samples, emb)))
    official_eval.n = (len(sc), len(labels))
    return fitb, auc


@torch.no_grad()
def own_eval(model, sampler, emb, n=4000, seed=123):
    """Validation with slot-preserving ("hard") negatives - closer to what the app needs."""
    g = torch.random.get_rng_state(), torch.cuda.get_rng_state() if DEV == "cuda" else None
    torch.manual_seed(seed)
    r, s, p = sampler.fitb(n)
    fb = model(emb[r], s, p).view(-1, 4)
    r1, s1, p1 = sampler.positive(n)
    r2, s2, p2 = sampler.negative(n)
    sc = torch.cat([model(emb[r1], s1, p1), model(emb[r2], s2, p2)]).float().cpu().numpy()
    torch.random.set_rng_state(g[0])
    if g[1] is not None:
        torch.cuda.set_rng_state(g[1])
    return float((fb.argmax(1) == 0).float().mean()), float(roc_auc_score([1] * n + [0] * n, sc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.02, help="gaussian noise on embeddings (robustness to phone photos)")
    ap.add_argument("--steps_per_epoch", type=int, default=150)
    ap.add_argument("--out", default=str(ROOT / "models" / "compat.pt"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pseudo", action="store_true", help="also train on title-only outfits (see embed_missing_text.py)")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    row, emb, _ = load_embeddings(a.pseudo)
    emb = emb.to(DEV)
    cat2slot = load_category_slots(DATA / "polyvore" / "category_id.txt")
    tr, so1 = load_outfits("train", row, cat2slot)
    va, so2 = load_outfits("valid", row, cat2slot)
    te, so3 = load_outfits("test", row, cat2slot)
    slot_of = {**so1, **so2, **so3}
    lookup = torch.full((len(emb),), OTHER, dtype=torch.long)
    for iid, s in slot_of.items():
        if iid in row:
            lookup[row[iid]] = s
    S_tr, S_va = Sampler(tr, lookup), Sampler(va, lookup)

    model = OutfitCompat(in_dim=emb.shape[1], d=a.d, layers=a.layers, dropout=a.dropout).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05)
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for _ in range(a.steps_per_epoch):
            parts = [S_tr.positive(a.bs), S_tr.negative(a.bs), S_tr.fitb(a.bs // 2)]
            r, s, p = (torch.cat(x) for x in zip(*parts))
            e = emb[r]
            e = F.normalize(e + a.noise * torch.randn_like(e), dim=-1)
            logit = model(e, s, p)
            y = torch.cat([torch.ones(a.bs), torch.zeros(a.bs)]).to(DEV)
            loss_cp = F.binary_cross_entropy_with_logits(logit[:2 * a.bs], y)
            lf = logit[2 * a.bs:].view(-1, 4)
            loss_fb = F.cross_entropy(lf, torch.zeros(len(lf), dtype=torch.long, device=DEV))
            loss = loss_cp + loss_fb
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        model.eval()
        vf, va_auc = own_eval(model, S_va, emb)
        msg = f"ep {ep+1:02d} loss {tot/a.steps_per_epoch:.3f} | valid(hard) FITB {vf:.3f} AUC {va_auc:.3f}"
        if vf + va_auc > best:
            best = vf + va_auc
            tf, ta = official_eval(model, emb, row, slot_of)
            save(model, a.out, metrics=dict(valid_fitb=vf, valid_auc=va_auc, test_fitb=tf, test_auc=ta), args=vars(a))
            msg += f" | TEST(official, n={official_eval.n}) FITB {tf:.3f} AUC {ta:.3f}  *saved"
        print(msg, f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
