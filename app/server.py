"""Aura Styles - web server.  Run:  python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

Every visitor gets a private wardrobe keyed by a random id kept in their browser
(no accounts).  Wardrobes live on disk under data/users/<id>/.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aura import model as compat
from aura.encoder import load_encoder
from aura.stylist import Item, Stylist
from aura.taxonomy import SLOT_ID, SLOT_LABEL
from aura.vision import Vision

ROOT = Path(__file__).resolve().parents[1]
USERS = ROOT / "data" / "users"
DEMO = ROOT / "app" / "demo"
STATIC = ROOT / "app" / "static"
MAX_ITEMS, MAX_BYTES, MAX_USERS = 150, 15 * 1024 * 1024, 300
UID_RX = re.compile(r"^[a-f0-9]{32}$")
IID_RX = re.compile(r"^[a-f0-9]{12}$")

app = FastAPI(title="Aura Styles")
enc = load_encoder()
vision = Vision(enc)
stylist = Stylist(enc, compat.load(ROOT / "models" / "compat.pt", enc.device))
_locks: dict[str, threading.Lock] = {}


def udir(uid: str | None, create: bool = False) -> Path:
    if not uid or not UID_RX.match(uid):
        raise HTTPException(400, "missing or invalid user id")
    d = USERS / uid
    if create and not d.exists():
        USERS.mkdir(parents=True, exist_ok=True)
        if sum(1 for _ in USERS.iterdir()) >= MAX_USERS:  # keeps a public test link from filling the disk
            raise HTTPException(503, "this test server is full")
        d.mkdir(exist_ok=True)
    return d


def lock(uid: str) -> threading.Lock:
    return _locks.setdefault(uid, threading.Lock())


def load_meta(d: Path) -> list[dict]:
    f = d / "items.json"
    return json.loads(f.read_text("utf8")) if f.exists() else []


def save_meta(d: Path, meta: list[dict]):
    tmp = d / "items.json.tmp"
    tmp.write_text(json.dumps(meta), "utf8")
    tmp.replace(d / "items.json")


def public(meta):
    return dict(items=meta, slots=SLOT_LABEL, max=MAX_ITEMS)


@app.get("/api/wardrobe")
def wardrobe(x_aura_user: str = Header(None)):
    return public(load_meta(udir(x_aura_user)))


@app.post("/api/items")
def add_items(files: list[UploadFile] = File(...), x_aura_user: str = Header(None)):
    d = udir(x_aura_user, create=True)
    added, errors = [], []
    for f in files:
        data = f.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            errors.append(f"{f.filename}: larger than 15 MB")
            continue
        try:
            a = vision.analyse(data)
        except Exception:
            errors.append(f"{f.filename}: not a readable image")
            continue
        iid = uuid.uuid4().hex[:12]
        with lock(x_aura_user):
            meta = load_meta(d)
            if len(meta) >= MAX_ITEMS:
                errors.append(f"wardrobe is full ({MAX_ITEMS} items)")
                break
            (d / f"{iid}.png").write_bytes(a["png"])
            np.save(d / f"{iid}.npy", a["emb"])
            item = dict(id=iid, name=a["name"], type=a["type"], slot=a["slot"], style=a["style"],
                        color=a["color"], colors=a["colors"], added=int(time.time()))
            meta.append(item)
            save_meta(d, meta)
        added.append(item)
    return dict(added=added, errors=errors, **public(load_meta(d)))


class Patch(BaseModel):
    slot: str | None = None
    name: str | None = None


@app.patch("/api/items/{iid}")
def patch_item(iid: str, p: Patch, x_aura_user: str = Header(None)):
    d = udir(x_aura_user)
    with lock(x_aura_user):
        meta = load_meta(d)
        for it in meta:
            if it["id"] == iid:
                if p.slot is not None:
                    if p.slot not in SLOT_LABEL:
                        raise HTTPException(400, "unknown slot")
                    it["slot"] = p.slot
                if p.name is not None:
                    it["name"] = p.name.strip()[:60] or it["name"]
                save_meta(d, meta)
                return it
    raise HTTPException(404, "no such item")


@app.delete("/api/items/{iid}")
def delete_item(iid: str, x_aura_user: str = Header(None)):
    d = udir(x_aura_user)
    if not IID_RX.match(iid):
        raise HTTPException(400, "bad id")
    with lock(x_aura_user):
        meta = [it for it in load_meta(d) if it["id"] != iid]
        if d.exists():
            save_meta(d, meta)
        for ext in ("png", "npy"):
            (d / f"{iid}.{ext}").unlink(missing_ok=True)
    return public(meta)


@app.delete("/api/wardrobe")
def clear(x_aura_user: str = Header(None)):
    d = udir(x_aura_user)
    with lock(x_aura_user):
        shutil.rmtree(d, ignore_errors=True)
    return public([])


@app.post("/api/demo")
def demo(x_aura_user: str = Header(None)):
    """Copy the bundled sample wardrobe (held-out Polyvore test items) into this visitor's wardrobe."""
    d = udir(x_aura_user, create=True)
    if not (DEMO / "items.json").exists():
        raise HTTPException(404, "demo wardrobe not built")
    with lock(x_aura_user):
        meta = load_meta(d)
        have = {it.get("demo") for it in meta}
        for it in json.loads((DEMO / "items.json").read_text("utf8")):
            if it["id"] in have or len(meta) >= MAX_ITEMS:
                continue
            iid = uuid.uuid4().hex[:12]
            for ext in ("png", "npy"):
                shutil.copyfile(DEMO / f"{it['id']}.{ext}", d / f"{iid}.{ext}")
            meta.append({**it, "id": iid, "demo": it["id"], "added": int(time.time())})
        save_meta(d, meta)
    return public(meta)


@app.get("/img/{uid}/{iid}.png")
def img(uid: str, iid: str):
    if not (UID_RX.match(uid) and IID_RX.match(iid)):
        raise HTTPException(404)
    f = USERS / uid / f"{iid}.png"
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


class SuggestReq(BaseModel):
    occasion: str = ""
    weather: str = ""
    anchors: list[str] = []
    n: int = 6
    accessories: bool = True
    seed: int = 0


@app.post("/api/suggest")
def suggest(r: SuggestReq, x_aura_user: str = Header(None)):
    d = udir(x_aura_user)
    items = []
    for it in load_meta(d):
        f = d / f"{it['id']}.npy"
        if f.exists() and it["slot"] in SLOT_ID:
            items.append(Item(it["id"], it["slot"], it.get("color", "gray"), np.load(f)))
    return stylist.suggest(items, n=max(1, min(r.n, 12)), occasion=r.occasion[:120], weather=r.weather,
                           anchors=r.anchors[:8], accessories=r.accessories, seed=r.seed)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
def health():
    return Response("ok")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
