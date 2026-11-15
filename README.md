# Aura Styles

Upload photos of your clothes, get outfits that go together.  A test build: a
learned outfit-compatibility model behind a small web app you can share with a link.

## Run it

```
run_local.bat          ->  http://localhost:8000  (this PC only)
share_link.bat         ->  temporary public https link for friends (needs cloudflared)
```

The first start takes ~20 s while the models load.  Each visitor gets a private
wardrobe (a random id stored in their browser, no accounts); photos and
embeddings are kept under `data/users/` on this PC.  "Clear all" deletes them.

## Deploy on Streamlit Community Cloud (GitHub → Streamlit)

`streamlit_app.py` is the same stylist behind a Streamlit UI with **username +
password sign-in**; every account has its own private wardrobe.

1. Create an empty repository on GitHub (private is fine), then from this folder:
   ```
   git remote add origin https://github.com/<you>/aura-styles.git
   git push -u origin main
   ```
2. On https://share.streamlit.io choose **Create app → Deploy a public app from
   GitHub**, pick the repository, branch `main`, main file `streamlit_app.py`.
3. The first boot takes several minutes (it installs PyTorch and downloads the
   ~800 MB FashionSigLIP weights).  Share the `*.streamlit.app` address.

Run it locally first with `python -m streamlit run streamlit_app.py`.

Things to know about the free tier:

* **Storage is temporary.**  Accounts and photos live on the app's local disk,
  which Streamlit wipes whenever the app restarts, is redeployed, or wakes from
  sleep (apps sleep after a period without visitors).  Fine for a test session; for
  lasting wardrobes the store (`aura/store.py`) needs an external database/bucket.
* **Memory is tight.**  The app uses about 2 GB of the ~2.7 GB limit, and it is
  CPU-only: about 1 s per uploaded photo, well under 1 s per suggestion.
* Passwords are stored as salted PBKDF2 hashes, but this is a test build: tell
  friends to use a throwaway password.

## How it works

```
photo ──► background removal (rembg) ──► cut-out on white
      ──► Marqo-FashionSigLIP image embedding (768-d)
      ──► zero-shot garment type → slot (top / bottom / dress / shoes / bag …),
          dominant colour (k-means in Lab), style tag

wardrobe ─► enumerate cores:  (top + bottom | dress) + shoes
         ─► score every core with the compatibility transformer   (learned on Polyvore)
         ─► + occasion text match, weather, light colour-harmony prior
         ─► grow the best cores with outerwear / bag / accessories while the score rises
         ─► diverse top-6 (maximal marginal relevance + item-reuse penalty)
```

| file | role |
|---|---|
| `aura/encoder.py` | FashionSigLIP image + text encoder (shared by training and the app) |
| `aura/vision.py` | upload pipeline: cut-out, embedding, auto-tagging, colour |
| `aura/model.py` | `OutfitCompat` – set transformer that outputs one compatibility logit per outfit |
| `aura/stylist.py` | outfit generation and ranking |
| `aura/taxonomy.py` | outfit slots; mapping of Polyvore's 380 categories onto them |
| `app/server.py`, `app/static/index.html` | FastAPI server and the single-page UI |
| `streamlit_app.py`, `aura/store.py` | Streamlit UI with sign-in; accounts and wardrobes on disk |
| `training/` | reproducible pipeline (below) |

## The model, and how it relates to the paper

Han et al. 2017 (*Learning Fashion Compatibility with Bidirectional LSTMs*) treat an
outfit as a sequence, run a Bi-LSTM over Inception-V3 features and add a
visual-semantic embedding of the item titles.  Aura Styles keeps the paper's task
definitions and its dataset/benchmarks but swaps both halves for their modern
counterparts:

* **Item features** – Inception-V3 + a small learned title embedding →
  **Marqo-FashionSigLIP**, a fashion-tuned image/text model.  It already supplies the
  joint visual-semantic space (the paper's "VSE" half), which is what powers
  occasion prompts ("rooftop dinner") and auto-tagging.
* **Compatibility** – Bi-LSTM over a fixed item order → an order-invariant
  **set transformer** with an `[OUTFIT]` token (as in OutfitTransformer, Sarkar et
  al. 2022).  A wardrobe has no canonical order, and the transformer scores a whole
  outfit in one pass.
* **Training signal** – real Polyvore outfits vs. corrupted ones (BCE) plus a 4-way
  fill-in-the-blank ranking loss.  Negatives are mostly *slot-preserving* (a top is
  swapped for another top) so the model learns style and colour rather than "two
  pairs of shoes is wrong"; the app's templates already guarantee structure.
  Outfits are randomly sub-sampled so partial outfits (3–5 pieces) score sensibly.

### Results on the paper's own test files

| | fill-in-the-blank accuracy | compatibility AUC |
|---|---|---|
| Bi-LSTM + VSE (Han et al. 2017, as published) | 68.6 % | 0.90 |
| **Aura Styles** (FashionSigLIP + set transformer) | **75.7 %** | **0.917** |
| Aura Styles, harder validation (same-category substitutes) | 66.1 % | 0.856 |

Caveats, so the comparison is read fairly: images come from the `Marqo/polyvore`
mirror, which holds 94,095 of the 164,379 items.  7,010 of the compatibility
samples could be scored (nearly all), but only 1,352 of the 3,076 fill-in-the-blank questions
have images for all four candidate answers, so the FITB number is on that subset.
Training used the ~55 % of training items that have images.  A variant that also
trained on title-only outfits (`--pseudo`) scored 76.0 % / 0.912 on the test files
but worse on the harder validation (64.7 % / 0.833), so it is not the shipped model.

The TensorFlow-1 checkpoint of the original Bi-LSTM (`model.ckpt-34865`) is not used
at runtime: it needs Inception-V3 features, a fixed item order and a reference pool
of item embeddings to score against, and the new model outperforms its published
numbers.

## Retrain

```
python training/embed_polyvore.py        # ~2.5 GB download, images are embedded then deleted
python training/embed_missing_text.py    # optional: titles of train items that have no image
python training/train_compat.py [--pseudo]
python training/build_demo.py            # sample wardrobe from held-out test items
```

## Known limits

* Polyvore is mostly women's fashion from 2016-17; menswear works through the
  encoder's general fashion knowledge but has seen far fewer examples.
* One garment per photo.  Photos of a person wearing a full outfit are treated
  as a single item.
* Colour names come from the dominant colour, so prints are described roughly.
* The public link is a tunnel to this PC: it is only up while `share_link.bat` runs.
