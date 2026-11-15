"""Shared image/text encoder: Marqo-FashionSigLIP (ViT-B/16 SigLIP fine-tuned on fashion).

Used both offline (embedding the Polyvore training items) and online (embedding
wardrobe uploads and occasion prompts), so both live in the same vector space.
"""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

MODEL_ID = "hf-hub:Marqo/marqo-fashionSigLIP"
DIM = 768


class Encoder:
    def __init__(self, device: str | None = None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL_ID)
        self.model = model.eval().to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(MODEL_ID)
        self.lock = threading.Lock()

    @property
    def dtype(self):
        return torch.float16 if self.device == "cuda" else torch.float32

    @torch.no_grad()
    def embed_tensor(self, x: torch.Tensor) -> torch.Tensor:
        with self.lock:
            f = self.model.encode_image(x.to(self.device, self.dtype))
        return F.normalize(f.float(), dim=-1)

    def embed_images(self, pil_images) -> torch.Tensor:
        return self.embed_tensor(torch.stack([self.preprocess(im) for im in pil_images]))

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> torch.Tensor:
        tok = self.tokenizer(texts).to(self.device)
        with self.lock:
            f = self.model.encode_text(tok)
        return F.normalize(f.float(), dim=-1)


_ENC = None


def load_encoder() -> Encoder:
    global _ENC
    if _ENC is None:
        _ENC = Encoder()
    return _ENC
