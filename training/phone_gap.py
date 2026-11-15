"""How product-like are phone-photo embeddings?  (the outfit model only ever saw product shots)

For each phone test photo: mean cosine similarity to its 10 nearest Polyvore product
embeddings (higher = closer to the training distribution), and whether those
neighbours are the right kind of garment (kNN slot accuracy).
    python training/phone_gap.py raw silueta ...
"""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.phone_eval import DEV, phone_data, product_data  # noqa: E402

PX, Py = product_data()
print(f"{'variant':22} {'nn-cos':>7} {'knn-slot-acc':>12}")
for v in sys.argv[1:]:
    X, y, split, _ = phone_data(v)
    t = torch.from_numpy(split == "test").to(DEV)
    sim, idx = (X[t] @ PX.T).topk(10, dim=1)
    vote = torch.mode(Py[idx], dim=1).values
    print(f"{v:22} {sim.mean().item():7.3f} {(vote == y[t]).float().mean().item():12.3f}")
# reference: product shots against other product shots
p = torch.randperm(len(PX), device=DEV)[:372]
sim, _ = (PX[p] @ PX.T).topk(11, dim=1)
print(f"{'(product vs product)':22} {sim[:, 1:].mean().item():7.3f}")
