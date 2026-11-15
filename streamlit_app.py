"""Aura Styles - Streamlit version (deployable on Streamlit Community Cloud).

    streamlit run streamlit_app.py

Each friend signs in with a username + password and gets a private wardrobe.
"""
from __future__ import annotations

import streamlit as st

from aura import store
from aura.taxonomy import SLOT_ID, SLOT_LABEL

st.set_page_config(page_title="Aura Styles", page_icon="✨", layout="wide")

OCCASIONS = ["Casual day", "Work", "Date night", "Party", "Formal event", "Gym", "Beach day", "Cozy winter"]
WEATHER = {"Any": "", "Hot": "hot", "Mild": "mild", "Cold": "cold"}

st.markdown("""
<style>
.block-container{padding-top:2.2rem;max-width:1200px}
h1{font-family:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;letter-spacing:.01em}
[data-testid="stImage"] img{background:#fff;border-radius:12px;padding:6px;object-fit:contain;aspect-ratio:1}
.aura-score{font:600 1.05rem Georgia,serif}
.aura-why{color:gray;font-size:.9rem;margin:.1rem 0 .4rem}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the stylist model (first start takes a minute)…")
def engine():
    from aura import model as compat
    from aura.encoder import load_encoder
    from aura.stylist import Stylist
    from aura.vision import Vision
    enc = load_encoder()
    return Vision(enc), Stylist(enc, compat.load(store.ROOT / "models" / "compat.pt", enc.device))


# ------------------------------------------------------------------ sign in
def sign_in():
    st.title("Aura Styles")
    st.write("Add photos of your clothes. Get outfits that actually go together.")
    left, _ = st.columns([1, 1.4])
    with left:
        mode = st.segmented_control("Account", ["Sign in", "Create account"], default="Sign in",
                                    label_visibility="collapsed") or "Sign in"
        with st.form("login"):
            username = st.text_input("Username", max_chars=20)
            password = st.text_input("Password", type="password", max_chars=100)
            if st.form_submit_button(mode, type="primary"):
                user, err = store.login(username, password, create=(mode == "Create account"))
                if user:
                    st.session_state.user = user
                    st.rerun()
                st.error(err)
        st.caption("Test build. Your wardrobe is private to your username. Use a throwaway password — "
                   "not one you use anywhere else.")


if "user" not in st.session_state:
    sign_in()
    st.stop()

user = st.session_state.user
vision, stylist = engine()
items = store.items(user)
by_id = {it["id"]: it for it in items}

# ------------------------------------------------------------------ header
head, acct = st.columns([4, 1], vertical_alignment="bottom")
head.title("Aura Styles")
with acct:
    st.caption(f"Signed in as **{user}**")
    if st.button("Sign out"):
        st.session_state.clear()
        st.rerun()

tab_outfits, tab_wardrobe = st.tabs(["Outfits", "Wardrobe"])

# ------------------------------------------------------------------ wardrobe
with tab_wardrobe:
    st.caption(f"{len(items)} item{'s' if len(items) != 1 else ''} in your wardrobe")
    up_col, demo_col = st.columns([3, 1], vertical_alignment="bottom")
    with up_col:
        with st.form("upload", clear_on_submit=True, border=False):
            files = st.file_uploader("Add clothes — one item per photo, laid flat or on a hanger works best",
                                     type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)
            go_up = st.form_submit_button("Add to wardrobe", type="primary")
        if go_up and files:
            bar = st.progress(0.0, "Analysing…")
            skipped = []
            for k, f in enumerate(files):
                try:
                    if not store.add(user, vision.analyse(f.getvalue())):
                        skipped.append(f"wardrobe is full ({store.MAX_ITEMS} items)")
                        break
                except Exception:
                    skipped.append(f"{f.name}: not a readable image")
                bar.progress((k + 1) / len(files), f"Analysing {k + 1} of {len(files)}…")
            if skipped:
                st.session_state.flash = "Some photos were skipped — " + "; ".join(skipped)
            st.rerun()
    with demo_col:
        if st.button("Try a sample wardrobe", use_container_width=True):
            store.add_demo(user)
            st.rerun()
    if msg := st.session_state.pop("flash", None):
        st.warning(msg)

    if not items:
        st.info("Nothing here yet. Add a few tops, bottoms and shoes — or try the sample wardrobe.")
    else:
        used = [s for s in SLOT_LABEL if any(it["slot"] == s for it in items)]
        show = st.pills("Show", ["All"] + [SLOT_LABEL[s] for s in used], default="All",
                        label_visibility="collapsed") or "All"
        order = list(SLOT_LABEL)
        shown = sorted((it for it in items if show == "All" or SLOT_LABEL.get(it["slot"]) == show),
                       key=lambda it: (order.index(it["slot"]) if it["slot"] in order else 99, -it["added"]))
        st.caption("Tick “Build around” to force a piece into every outfit. Wrong category? Fix it in the dropdown.")
        cols = st.columns(6)
        for k, it in enumerate(shown):
            with cols[k % 6]:
                st.image(store.png_path(user, it["id"]), use_container_width=True)
                st.caption(it["name"] + ("  \n:orange[Not sure — please check the category]" if it.get("unsure") else ""))
                slot = st.selectbox("Category", order, index=order.index(it["slot"]) if it["slot"] in order else 0,
                                    format_func=SLOT_LABEL.get, key=f"slot_{it['id']}", label_visibility="collapsed")
                if slot != it["slot"]:
                    store.set_slot(user, it["id"], slot)
                    st.rerun()
                st.checkbox("Build around", key=f"pin_{it['id']}")
                if st.button("Remove", key=f"del_{it['id']}"):
                    store.delete(user, it["id"])
                    st.rerun()
        st.divider()
        with st.expander("Delete my data"):
            if st.button("Clear my wardrobe"):
                store.clear(user)
                st.rerun()
            if st.button("Delete my account and all photos"):
                store.delete_account(user)
                st.session_state.clear()
                st.rerun()

# ------------------------------------------------------------------ outfits
with tab_outfits:
    if len(items) < 2:
        st.info("Add at least two items in the Wardrobe tab to get outfit ideas.")
    else:
        occ = st.pills("Occasion", OCCASIONS)
        c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="bottom")
        free = c1.text_input("…or describe it", placeholder="rooftop dinner, job interview, rainy day errands",
                             max_chars=100)
        weather = c2.segmented_control("Weather", list(WEATHER), default="Any") or "Any"
        acc = c3.toggle("Add bags & accessories", value=True)
        pins = [it["id"] for it in items if st.session_state.get(f"pin_{it['id']}")]
        if pins:
            st.caption("Building around: " + ", ".join(by_id[i]["name"] for i in pins))
        b1, b2, _ = st.columns([1.6, 1, 4.4])
        if b1.button("Suggest outfits", type="primary"):
            st.session_state.seed, st.session_state.asked = 0, True
        if b2.button("Shuffle", disabled=not st.session_state.get("asked")):
            st.session_state.seed = st.session_state.get("seed", 0) + 1

        if st.session_state.get("asked"):
            from aura.stylist import Item
            pool = [Item(it["id"], it["slot"], it.get("color", "gray"), store.emb(user, it["id"]))
                    for it in items if it["slot"] in SLOT_ID]
            with st.spinner("Styling…"):
                res = stylist.suggest(pool, n=6, occasion=(free.strip() or occ or ""), weather=WEATHER[weather],
                                      anchors=pins, accessories=acc, seed=st.session_state.get("seed", 0))
            if res["note"]:
                st.warning(res["note"])
            cols = st.columns(3)
            for k, o in enumerate(res["outfits"]):
                with cols[k % 3].container(border=True):
                    its = [by_id[i] for i in o["items"] if i in by_id]
                    per_row = 2 if len(its) in (2, 4) else 3
                    for r in range(0, len(its), per_row):
                        row = st.columns(per_row)
                        for c, it in zip(row, its[r:r + per_row]):
                            c.image(store.png_path(user, it["id"]), use_container_width=True, caption=it["name"])
                    st.markdown(f"<div class='aura-score'>Look {k + 1} · {o['match']}% match</div>",
                                unsafe_allow_html=True)
                    st.markdown(f"<div class='aura-why'>{o['why']}</div>", unsafe_allow_html=True)
