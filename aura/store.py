"""Accounts and per-user wardrobes on disk (used by the Streamlit app).

data/accounts/<username>/account.json   salted PBKDF2 password hash
data/accounts/<username>/items.json     wardrobe metadata
data/accounts/<username>/<id>.png|.npy  cut-out and embedding of each item
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACCOUNTS = ROOT / "data" / "accounts"
DEMO = ROOT / "app" / "demo"
USER_RX = re.compile(r"^[a-z0-9_]{3,20}$")
MAX_ITEMS, MAX_USERS = 150, 200


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def login(username: str, password: str, create: bool) -> tuple[str | None, str]:
    """-> (normalised username or None, error message)"""
    u = username.strip().lower()
    if not USER_RX.match(u):
        return None, "Usernames are 3–20 characters: letters, numbers or underscore."
    d = ACCOUNTS / u
    f = d / "account.json"
    if create:
        if f.exists():
            return None, "That username is taken. Sign in instead, or pick another."
        if len(password) < 6:
            return None, "Pick a password with at least 6 characters."
        ACCOUNTS.mkdir(parents=True, exist_ok=True)
        if sum(1 for _ in ACCOUNTS.iterdir()) >= MAX_USERS:
            return None, "This test app is full."
        d.mkdir(exist_ok=True)
        salt = os.urandom(16)
        f.write_text(json.dumps(dict(salt=salt.hex(), hash=_hash(password, salt), created=int(time.time()))))
        return u, ""
    if not f.exists():
        return None, "No account with that username yet. Switch to “Create account”."
    a = json.loads(f.read_text())
    if not hmac.compare_digest(a["hash"], _hash(password, bytes.fromhex(a["salt"]))):
        return None, "Wrong password."
    return u, ""


def _dir(user: str) -> Path:
    assert USER_RX.match(user)
    return ACCOUNTS / user


def items(user: str) -> list[dict]:
    f = _dir(user) / "items.json"
    return json.loads(f.read_text("utf8")) if f.exists() else []


def _save(user: str, meta: list[dict]):
    tmp = _dir(user) / "items.json.tmp"
    tmp.write_text(json.dumps(meta), "utf8")
    tmp.replace(_dir(user) / "items.json")


def add(user: str, a: dict, demo: str | None = None) -> bool:
    meta = items(user)
    if len(meta) >= MAX_ITEMS:
        return False
    iid = uuid.uuid4().hex[:12]
    (_dir(user) / f"{iid}.png").write_bytes(a["png"])
    np.save(_dir(user) / f"{iid}.npy", a["emb"])
    it = dict(id=iid, name=a["name"], type=a["type"], slot=a["slot"], style=a["style"],
              color=a["color"], colors=a["colors"], added=int(time.time()))
    if a.get("unsure"):
        it["unsure"] = True
    if demo:
        it["demo"] = demo
    _save(user, meta + [it])
    return True


def add_demo(user: str) -> int:
    have = {it.get("demo") for it in items(user)}
    n = 0
    for it in json.loads((DEMO / "items.json").read_text("utf8")):
        if it["id"] in have:
            continue
        a = dict(it, png=(DEMO / f"{it['id']}.png").read_bytes(), emb=np.load(DEMO / f"{it['id']}.npy"))
        if not add(user, a, demo=it["id"]):
            break
        n += 1
    return n


def set_slot(user: str, iid: str, slot: str):
    meta = items(user)
    for it in meta:
        if it["id"] == iid:
            it["slot"] = slot
            it.pop("unsure", None)  # the user has confirmed the category
    _save(user, meta)


def delete(user: str, iid: str):
    _save(user, [it for it in items(user) if it["id"] != iid])
    for ext in ("png", "npy"):
        (_dir(user) / f"{iid}.{ext}").unlink(missing_ok=True)


def clear(user: str):
    for f in _dir(user).glob("*"):
        if f.name != "account.json":
            f.unlink()


def delete_account(user: str):
    shutil.rmtree(_dir(user), ignore_errors=True)


def png_path(user: str, iid: str) -> str:
    return str(_dir(user) / f"{iid}.png")


def emb(user: str, iid: str) -> np.ndarray:
    return np.load(_dir(user) / f"{iid}.npy")
