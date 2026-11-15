"""Embed the public phone-photo clothing dataset (alexeygrigorev/clothing-dataset-small)
under different preprocessing variants, so phone-photo handling can be measured.

    python training/phone_embed.py raw silueta u2net isnet-general-use ...

"raw" embeds the untouched photo; any other name is a rembg matting model whose
cut-out is placed on white (what the app does).  Output: data/phone/emb_<variant>.npz
with ids, split, label (dataset class), emb [N,768], and mask_ok (cut-out accepted).
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.encoder import load_encoder  # noqa: E402
from aura.vision import Vision  # noqa: E402

SRC = ROOT / "data" / "phone" / "clothing-dataset-small"


def main():
    enc = load_encoder()
    v = Vision(enc)
    files = [(s, f.parent.name, f) for s in ("train", "validation", "test") for f in sorted((SRC / s).glob("*/*"))]
    for variant in sys.argv[1:]:
        out = ROOT / "data" / "phone" / f"emb_{variant}.npz"
        if out.exists():
            print("skip", variant)
            continue
        if variant != "raw":
            from rembg import new_session
            v._rembg = new_session(variant)
        embs, ok, t0 = [], [], time.time()
        batch = []

        def flush():
            if batch:
                embs.append(enc.embed_images(batch).cpu().numpy().astype(np.float16))
                batch.clear()

        for k, (_, _, f) in enumerate(files):
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            im.thumbnail((1024, 1024))
            if variant == "raw":
                batch.append(im)
                ok.append(True)
            else:
                rgba = v.cutout(im)
                ok.append(np.array(rgba)[..., 3].min() < 255)
                batch.append(v.on_white(rgba))
            if len(batch) == 64:
                flush()
            if k % 500 == 499:
                print(f"  {variant} {k+1}/{len(files)} {(k+1)/(time.time()-t0):.1f}/s", flush=True)
        flush()
        np.savez(out, ids=np.array([f.stem for _, _, f in files]), split=np.array([s for s, _, _ in files]),
                 label=np.array([l for _, l, _ in files]), emb=np.concatenate(embs), mask_ok=np.array(ok))
        print(f"{variant}: {len(files)} photos, cut-out accepted {np.mean(ok):.1%}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
