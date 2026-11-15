"""Step 1 - embed every Polyvore item image (and its title) with Marqo-FashionSigLIP.

Images come from the Hugging Face mirror `Marqo/polyvore`, whose item ids
("<set_id>_<index>") match the ids in the Maryland Polyvore json files used by
Han et al. 2017.  Shards are processed one at a time and deleted afterwards so
the raw images never need more than ~450 MB of disk.

Output: data/emb/shard_<k>.npz  (ids, img float16 [N,768], txt float16 [N,768], category, text)
"""
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aura.encoder import load_encoder  # noqa: E402

OUT = ROOT / "data" / "emb"
RAW = ROOT / "data" / "raw"
N_SHARDS = 6
BATCH = 128


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # keep the raw images of a few held-out test outfits: they become the app's demo wardrobe
    import json
    demo_sets = {o["set_id"] for o in json.load(open(ROOT / "data" / "polyvore" / "test_no_dup.json"))[:80]}
    demo_dir = ROOT / "data" / "demo_raw"
    demo_dir.mkdir(exist_ok=True)
    enc = load_encoder()
    pool = ThreadPoolExecutor(8)

    def prep(b):
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:
            im = Image.new("RGB", (224, 224), "white")
        return enc.preprocess(im)

    for k in range(N_SHARDS):
        dst = OUT / f"shard_{k}.npz"
        if dst.exists():
            print("skip", dst.name, flush=True)
            continue
        fn = f"data/data-{k:05d}-of-{N_SHARDS:05d}.parquet"
        t0 = time.time()
        path = hf_hub_download("Marqo/polyvore", fn, repo_type="dataset", local_dir=RAW)
        pf = pq.ParquetFile(path)
        print(f"shard {k}: {pf.metadata.num_rows} rows, cols={pf.schema_arrow.names}, dl {time.time()-t0:.0f}s", flush=True)
        ids, cats, texts, img_e, txt_e = [], [], [], [], []
        done = 0
        for rb in pf.iter_batches(batch_size=BATCH):
            d = rb.to_pydict()
            imgs = [x["bytes"] for x in d["image"]]
            x = torch.stack(list(pool.map(prep, imgs)))
            img_e.append(enc.embed_tensor(x).cpu().numpy().astype(np.float16))
            t = [s or "" for s in d["text"]]
            txt_e.append(enc.embed_text(t).cpu().numpy().astype(np.float16))
            for iid, b in zip(d["item_ID"], imgs):
                if iid.split("_")[0] in demo_sets:
                    (demo_dir / f"{iid}.jpg").write_bytes(b)
            ids += d["item_ID"]
            cats += d["category"]
            texts += t
            done += len(imgs)
            if done % (BATCH * 20) == 0:
                print(f"  {done} items  {done/(time.time()-t0):.0f}/s", flush=True)
        np.savez(dst, ids=np.array(ids), img=np.concatenate(img_e), txt=np.concatenate(txt_e),
                 category=np.array(cats), text=np.array(texts))
        del pf
        os.remove(path)
        print(f"shard {k} done: {len(ids)} items in {time.time()-t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
