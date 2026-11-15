"""Outfit generation from a wardrobe.

1. enumerate "core" outfits from templates:  (top + bottom | dress) + shoes
2. score every core with the learned compatibility transformer
3. add context terms: occasion text match, weather, a light colour-harmony prior
4. grow the best cores with outerwear / bag / accessories while the score improves
5. pick a diverse top-N (maximal marginal relevance)
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

import numpy as np
import torch

from .encoder import Encoder
from .model import OutfitCompat
from .taxonomy import DISPLAY_ORDER, SLOT_ID
from .vision import hue_family

MAX_CORES = 6000
KEEP_CORES = 48
OPTIONAL = ["outerwear", "bag", "hat", "eyewear", "scarf", "belt", "watch", "jewelry", "accessory"]
# weights are in units of "standard deviations of the compatibility score across this wardrobe's outfits"
W_OCC, W_WEATHER, W_COLOR = 1.2, 1.0, 0.5


# the UI's occasion chips, expanded into richer prompts for the text encoder
PRESETS = {
    "casual day": "a relaxed casual everyday weekend look",
    "work": "the office, professional business casual workwear",
    "date night": "a romantic date night, chic evening look",
    "party": "a night out party, glamorous going-out look",
    "formal event": "a formal event, elegant black tie evening wear",
    "gym": "the gym, athletic workout sportswear",
    "beach day": "a beach day, light summer vacation wear",
    "cozy winter": "a cold winter day, warm cozy knitwear and layers",
}


@dataclass
class Item:
    id: str
    slot: str
    color: str
    emb: np.ndarray


class Stylist:
    def __init__(self, enc: Encoder, model: OutfitCompat):
        self.enc, self.model = enc, model
        self.dev = next(model.parameters()).device
        self.warm = enc.embed_text(["warm winter clothing for cold weather"])[0]
        self.cool = enc.embed_text(["light summer clothing for hot weather"])[0]

    # ---- learned compatibility ----
    @torch.no_grad()
    def logits(self, E: torch.Tensor, S: torch.Tensor, combos: list[tuple[int, ...]]) -> np.ndarray:
        if not combos:
            return np.zeros(0, np.float32)
        n = max(len(c) for c in combos)
        idx = torch.zeros(len(combos), n, dtype=torch.long)
        pad = torch.ones(len(combos), n, dtype=torch.bool)
        for b, c in enumerate(combos):
            idx[b, :len(c)] = torch.tensor(c)
            pad[b, :len(c)] = False
        idx, pad = idx.to(self.dev), pad.to(self.dev)
        out = []
        for i in range(0, len(combos), 2048):
            j = idx[i:i + 2048]
            out.append(self.model(E[j], S[j], pad[i:i + 2048]))
        return torch.cat(out).float().cpu().numpy()

    # ---- hand-written priors ----
    @staticmethod
    def color_penalty(items: list[Item], combo) -> float:
        fams = {hue_family(items[i].color, items[i].slot) for i in combo
                if items[i].slot not in ("jewelry", "watch", "eyewear")}
        fams.discard(None)
        return max(0, len(fams) - 2) * 1.0 + (0.25 if len(fams) == 2 and not _harmonious(fams) else 0.0)

    def suggest(self, items: list[Item], n=6, occasion: str = "", weather: str = "", anchors: list[str] = (),
                accessories=True, seed: int = 0) -> dict:
        rng = random.Random(seed)
        if len(items) < 2:
            return dict(outfits=[], note="Add at least two items to get outfit ideas.")
        E = torch.from_numpy(np.stack([it.emb for it in items])).to(self.dev)
        S = torch.tensor([SLOT_ID[it.slot] for it in items], device=self.dev)
        by = {}
        for i, it in enumerate(items):
            by.setdefault(it.slot, []).append(i)
        anchor_idx = [i for i, it in enumerate(items) if it.id in set(anchors)]
        a_slot = {}
        for i in anchor_idx:
            a_slot.setdefault(items[i].slot, []).append(i)

        # context scores per item (z-scored inside this wardrobe)
        def z(v):
            v = v.float().cpu().numpy()
            return (v - v.mean()) / (v.std() + 1e-6) if len(v) > 2 else np.zeros_like(v)

        occ = np.zeros(len(items), np.float32)
        if occasion.strip():
            q = PRESETS.get(occasion.strip().lower(), occasion.strip())
            t = self.enc.embed_text([f"an outfit for {q}", f"{q} clothing"]).mean(0)
            occ = z(E @ t)
        warm = z(E @ (self.warm - self.cool))
        w_sign = {"cold": 1.0, "hot": -1.0}.get(weather, 0.0)

        # ---- 1. cores ----
        def pick(slot):
            return a_slot[slot][:1] if slot in a_slot else by.get(slot, [])

        tops, bottoms, dresses, shoes = pick("top"), pick("bottom"), pick("dress"), pick("shoes")
        cores = []
        want_dress = "dress" in a_slot
        want_sep = "top" in a_slot or "bottom" in a_slot
        if not want_dress:
            cores += [(t, b) for t in tops for b in bottoms]
        if not want_sep or want_dress:
            cores += [(d,) for d in dresses]
        note = ""
        if not cores:  # wardrobe has no complete base yet: pair whatever exists
            base = tops + bottoms + dresses
            cores = [(i,) for i in base] or [(i,) for i in range(len(items))][:50]
            missing = [s for s, l in (("tops", tops), ("bottoms", bottoms)) if not l]
            note = f"Add some {' and '.join(missing) or 'more pieces'} for complete outfits."
        if shoes:
            cores = [c + (s,) for c in cores for s in shoes]
        elif not note:
            note = "Add shoes to complete your outfits."
        forced = tuple(i for i in anchor_idx if items[i].slot not in ("top", "bottom", "dress", "shoes"))
        cores = [c + forced for c in cores]
        cores = [c for c in cores if len(c) >= 2] or cores
        if len(cores) > MAX_CORES:
            cores = rng.sample(cores, MAX_CORES)

        mu, sd = 0.0, 1.0

        def total(combos, lg):
            out = np.empty(len(combos), np.float32)
            for k, c in enumerate(combos):
                ci = list(c)
                out[k] = ((lg[k] - mu) / sd + W_OCC * occ[ci].mean() + W_WEATHER * w_sign * warm[ci].mean()
                          - W_COLOR * self.color_penalty(items, c))
            return out

        lg = self.logits(E, S, cores)
        mu, sd = float(lg.mean()), float(lg.std()) + 1e-3
        sc = total(cores, lg)
        if seed:  # "shuffle": perturb scores a little so new looks surface
            sc = sc + np.array([rng.gauss(0, 0.35) for _ in sc], np.float32)
        order = np.argsort(-sc)[:KEEP_CORES]
        cand = [(cores[i], float(sc[i])) for i in order]

        # ---- 4. grow with optional pieces ----
        slots = [s for s in OPTIONAL if s in by and (accessories or s == "outerwear")]
        if weather == "hot":
            slots = [s for s in slots if s not in ("outerwear", "scarf")]
        for slot in slots:
            trial, owner = [], []
            for k, (c, _) in enumerate(cand):
                if len(c) >= 7 or any(items[i].slot == slot for i in c):
                    continue
                for j in by[slot]:
                    trial.append(c + (j,))
                    owner.append(k)
            if not trial:
                continue
            ts = total(trial, self.logits(E, S, trial))
            need = 0.15 if slot in ("outerwear", "bag") else 0.3
            if slot == "outerwear" and weather == "cold":
                need = -math.inf
            best = {}
            for t, k, s in zip(trial, owner, ts):
                if s > best.get(k, (None, -math.inf))[1]:
                    best[k] = (t, float(s))
            for k, (t, s) in best.items():
                if s - cand[k][1] > need:
                    cand[k] = (t, s if need > -math.inf else max(s, cand[k][1]))

        # ---- 5. diverse top-N ----
        chosen, used = [], {}
        pool = sorted(cand, key=lambda x: -x[1])
        while pool and len(chosen) < n:
            def mmr(x):
                if not chosen:
                    return x[1]
                core = lambda c: {i for i in c if items[i].slot in ("top", "bottom", "dress", "outerwear", "shoes")}
                ov = max(len(core(x[0]) & core(y[0])) / max(1, len(core(x[0]) | core(y[0]))) for y in chosen)
                reuse = sum(used.get(i, 0) for i in x[0]) / len(x[0])  # don't lean on one favourite piece
                return x[1] - 2.5 * ov - 0.8 * reuse
            b = max(pool, key=mmr)
            pool.remove(b)
            if any(set(b[0]) == set(y[0]) for y in chosen):
                continue
            chosen.append(b)
            for i in b[0]:
                if i not in anchor_idx:
                    used[i] = used.get(i, 0) + 1
        final_lg = self.logits(E, S, [c for c, _ in chosen])
        outfits = []
        for (c, s), l in zip(chosen, final_lg):
            ordered = sorted(c, key=lambda i: DISPLAY_ORDER.index(items[i].slot))
            outfits.append(dict(items=[items[i].id for i in ordered],
                                match=round(100 / (1 + math.exp(-float(l)))),
                                why=self.explain(items, ordered, occ, occasion)))
        outfits.sort(key=lambda o: -o["match"])
        return dict(outfits=outfits, note=note)

    @staticmethod
    def explain(items, combo, occ, occasion) -> str:
        cols = []
        for i in combo:
            if items[i].slot in ("jewelry", "watch", "eyewear"):
                continue
            if items[i].color not in cols:
                cols.append(items[i].color)
        fams = {hue_family(items[i].color, items[i].slot) for i in combo} - {None}
        if not fams:
            tone = "an all-neutral palette"
        elif len(fams) == 1:
            tone = f"neutrals with a {next(iter(fams))} accent"
        else:
            tone = "a " + " + ".join(sorted(fams)) + " colour story"
        txt = f"{', '.join(cols[:4])} — {tone}."
        if occasion.strip() and len(combo):
            fit = float(np.mean([occ[i] for i in combo]))
            txt += f" {'Strong' if fit > 0.5 else 'Good' if fit > 0 else 'Loose'} fit for “{occasion.strip()}”."
        return txt[0].upper() + txt[1:]


_WHEEL = ["red", "orange", "yellow", "green", "blue", "purple", "pink"]


def _harmonious(fams) -> bool:
    a, b = (_WHEEL.index(f) for f in fams)
    d = min(abs(a - b), len(_WHEEL) - abs(a - b))
    return d == 1 or d >= 3  # analogous or roughly complementary
