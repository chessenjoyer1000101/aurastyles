"""Outfit compatibility model.

A small set-transformer over frozen FashionSigLIP item embeddings (the modern
successor of the Bi-LSTM in Han et al. 2017, following OutfitTransformer,
Sarkar et al. 2022): each item becomes one token (projected image embedding +
slot embedding), a learned [OUTFIT] token attends over all of them, and an MLP
on that token outputs a single compatibility logit.  Unlike the Bi-LSTM it is
order-invariant, so it does not depend on Polyvore's item ordering.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .taxonomy import SLOTS


class OutfitCompat(nn.Module):
    def __init__(self, in_dim=768, d=256, layers=3, heads=4, ff=512, dropout=0.2, in_dropout=0.1):
        super().__init__()
        self.cfg = dict(in_dim=in_dim, d=d, layers=layers, heads=heads, ff=ff,
                        dropout=dropout, in_dropout=in_dropout)
        self.in_drop = nn.Dropout(in_dropout)
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d))
        self.slot = nn.Embedding(len(SLOTS), d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, emb, slots, pad_mask):
        """emb [B,N,in_dim], slots [B,N] long, pad_mask [B,N] bool (True = padding) -> logits [B]"""
        x = self.proj(self.in_drop(emb)) + self.slot(slots)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1)
        m = torch.cat([pad_mask.new_zeros(len(x), 1), pad_mask], 1)
        x = self.enc(x, src_key_padding_mask=m)
        return self.head(x[:, 0]).squeeze(-1)


def save(model: OutfitCompat, path, **extra):
    torch.save({"cfg": model.cfg, "state": model.state_dict(), **extra}, path)


def load(path, device="cpu") -> OutfitCompat:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = OutfitCompat(**ck["cfg"])
    m.load_state_dict(ck["state"])
    return m.eval().to(device)
