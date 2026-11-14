"""Aura Stylist test bench.

    pip install -e ".[vision,gui]"
    streamlit run app.py

Everything is stored locally in ./aura_data (photos + state.json).
"""
import os
import pathlib
import uuid

import streamlit as st

from aura_stylist import (CATEGORIES, AuraStylist, ClipBackend, Context, ItemClassifier,
                          Occasion, Reason)
from aura_stylist.colors import COLOR_RGB
from aura_stylist.knowledge import DRESS_CODE_FORMALITY, PATTERNS, STYLES

ROOT = pathlib.Path(os.environ.get("AURA_DATA", "aura_data"))
# Hosted mode: each tester gets a separate wardrobe, chosen by the name they enter.
HOSTED = (os.environ.get("AURA_HOSTED") == "1"
          or (pathlib.Path(__file__).parent / "hosted.flag").exists())
MODELS = {"General CLIP": "general", "Fashion-tuned (bigger download)": "fashion",
          "Off (I label items myself)": "none"}
REASONS = {"No particular reason": None, "The colours": Reason.COLORS,
           "Too formal": Reason.TOO_FORMAL, "Too casual": Reason.TOO_CASUAL,
           "Too warm": Reason.TOO_WARM, "Too cold": Reason.TOO_COLD,
           "I don't like one item": Reason.DISLIKE_ITEM, "Not my style": Reason.NOT_MY_STYLE}
FEATURE_LABELS = {"harmony": "Colour harmony", "formality": "Fits the occasion",
                  "weather": "Fits the weather", "style": "Style", "color_pref": "Colours you like",
                  "pattern": "Pattern balance", "novelty": "Rarely worn pieces"}

st.set_page_config(page_title="Aura Stylist test bench", layout="wide")

if HOSTED and "tester" not in st.session_state:
    st.title("Aura Stylist")
    st.write("Enter your name to open your wardrobe. Use the same name each time you come back.")
    who = st.text_input("Your name", max_chars=40)
    slug = "".join(ch for ch in who.strip().lower().replace(" ", "-") if ch.isalnum() or ch == "-")
    if st.button("Start", type="primary", disabled=not slug):
        st.session_state.tester = slug
        st.rerun()
    st.caption("This is a test version. Photos are kept on the test server and may be cleared "
               "when it restarts. Anyone who enters the same name sees the same wardrobe.")
    st.stop()
DATA = ROOT / st.session_state.tester if HOSTED else ROOT
PHOTOS, STATE = DATA / "photos", DATA / "state.json"
if HOSTED:      # the larger fashion checkpoint does not fit free hosting memory
    MODELS = {k: v for k, v in MODELS.items() if v != "fashion"}


# ---------------------------------------------------------------- state
@st.cache_resource(show_spinner="Loading the vision model (first time downloads it)...")
def get_classifier(kind: str) -> ItemClassifier:
    if kind == "none" or not ClipBackend.available():
        return ItemClassifier(backend=None)
    if kind == "fashion":
        return ItemClassifier(ClipBackend("hf-hub:Marqo/marqo-fashionSigLIP", pretrained=None))
    return ItemClassifier(ClipBackend())


if "stylist" not in st.session_state:
    st.session_state.stylist = AuraStylist.load(str(STATE)) if STATE.exists() else AuraStylist()
    st.session_state.up_n = 0
    st.session_state.recs = {}      # tab name -> (Recommendation, Context)
    st.session_state.acted = {}     # outfit key -> "saved" | "rejected"
s: AuraStylist = st.session_state.stylist


def persist():
    DATA.mkdir(parents=True, exist_ok=True)
    s.save(str(STATE))


def clear_results():
    st.session_state.recs = {}
    st.session_state.acted = {}


def label(item) -> str:
    color = item.main_color.name if item.main_color else ""
    return f"{color} {item.category}".strip()


def show_photo(item):
    if item.image_path and pathlib.Path(item.image_path).exists():
        try:
            st.image(item.image_path, width="stretch")
        except Exception:               # older Streamlit
            st.image(item.image_path, use_container_width=True)
    else:
        rgb = item.main_color.rgb if item.main_color else (200, 200, 200)
        st.markdown(f"<div style='height:110px;border-radius:6px;background:rgb{tuple(rgb)}'></div>",
                    unsafe_allow_html=True)


def say(text: str):
    """Show stylist text, right-to-left when the profile language is Arabic."""
    if s.profile.language == "ar":
        st.markdown(f"<div dir='rtl' style='text-align:right'>{text}</div>", unsafe_allow_html=True)
    else:
        st.write(text)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("Aura Stylist")
    st.caption(f"Wardrobe: {st.session_state.tester}" if HOSTED
               else "Local test bench for the AI module")
    lang = st.radio("Stylist language", ["English", "العربية"],
                    index=1 if s.profile.language == "ar" else 0, horizontal=True)
    new_lang = "ar" if lang == "العربية" else "en"
    if new_lang != s.profile.language:
        s.profile.language = new_lang
        clear_results()
        persist()

    model_name = st.radio("Photo recognition", list(MODELS))
    if MODELS[model_name] != "none" and not ClipBackend.available():
        st.warning('Vision packages are not installed, so categories must be picked by hand. '
                   'To enable recognition run:  pip install -e ".[vision]"')

    items = list(s.wardrobe.values())
    c1, c2, c3 = st.columns(3)
    c1.metric("Items", len(items))
    c2.metric("To confirm", len(s.items_needing_review()))
    c3.metric("Ratings", s.prefs.n_feedback)

    st.divider()
    st.caption("Not in this test bench: the chat assistant and human stylist booking. "
               "They are app features outside the AI module.")
    with st.expander("Delete my data"):
        st.write("Removes every photo, item, saved outfit and everything learned from your ratings.")
        sure = st.checkbox("Yes, delete everything")
        if st.button("Delete all data", disabled=not sure):
            s.erase_all()
            if PHOTOS.exists():
                for p in PHOTOS.iterdir():
                    p.unlink()
            clear_results()
            persist()
            st.rerun()


# ---------------------------------------------------------------- outfits
def context_inputs(prefix: str) -> Context:
    a, b, c, d = st.columns([2, 2, 2, 2])
    occ = a.selectbox("Occasion", [o.value for o in Occasion], key=f"{prefix}_occ")
    code = b.selectbox("Dress code (optional)", ["none"] + list(DRESS_CODE_FORMALITY),
                       key=f"{prefix}_code")
    use_weather = c.checkbox("I know the weather", key=f"{prefix}_wx")
    temp = cond = None
    if use_weather:
        temp = c.number_input("Temperature (°C)", -20.0, 50.0, 22.0, 1.0, key=f"{prefix}_temp")
        cond = d.selectbox("Conditions", ["clear", "rain", "snow"], key=f"{prefix}_cond")
    return Context(Occasion(occ), temp, cond, None if code == "none" else code)


def render_results(tab: str):
    if tab not in st.session_state.recs:
        return
    rec, ctx = st.session_state.recs[tab]
    for note in rec.notes:
        st.info(note)
    if not rec.outfits:
        st.warning("No outfit could be built. Add more items, or confirm the ones marked for review.")
    for n, o in enumerate(rec.outfits):
        with st.container(border=True):
            cols = st.columns(max(len(o.items), 5))
            for col, it in zip(cols, o.items):
                with col:
                    show_photo(it)
                    st.caption(label(it))
            say(o.explanation)
            with st.expander(f"Score {o.score:.2f}: how it was worked out"):
                for f, lab in FEATURE_LABELS.items():
                    st.progress(min(1.0, max(0.0, o.features[f])), text=f"{lab}: {o.features[f]:.2f}")
            state = st.session_state.acted.get(o.key)
            if state:
                st.success("Saved to your outfits." if state == "saved"
                           else "Rejected. It will not be suggested again.")
                continue
            a, b, c, d = st.columns([2, 3, 3, 2])
            if a.button("Like and save", key=f"{tab}_like{n}", type="primary"):
                s.like(o, ctx)
                st.session_state.acted[o.key] = "saved"
                persist()
                st.rerun()
            why = b.selectbox("If not, why?", list(REASONS), key=f"{tab}_why{n}")
            bad = None
            if REASONS[why] is Reason.DISLIKE_ITEM:
                bad = c.selectbox("Which item?", o.item_ids(), key=f"{tab}_bad{n}",
                                  format_func=lambda i: label(s.wardrobe[i]))
            if d.button("Reject", key=f"{tab}_no{n}"):
                s.dislike(o, ctx, REASONS[why], bad)
                st.session_state.acted[o.key] = "rejected"
                persist()
                st.rerun()


# ---------------------------------------------------------------- tabs
t_ward, t_prof, t_out, t_chal, t_ins, t_saved = st.tabs(
    ["Wardrobe", "My preferences", "Outfits", "Style challenge", "Gaps and style", "Saved outfits"])

with t_ward:
    st.subheader("Add clothes")
    st.caption("One garment per photo, on a plain background, works best.")
    files = st.file_uploader("Photos", type=["jpg", "jpeg", "png", "webp"],
                             accept_multiple_files=True, key=f"up{st.session_state.up_n}")
    if st.button("Add to wardrobe", disabled=not files, type="primary"):
        PHOTOS.mkdir(parents=True, exist_ok=True)
        clf = get_classifier(MODELS[model_name])
        bar = st.progress(0.0, text="Looking at your photos...")
        for n, f in enumerate(files):
            iid = uuid.uuid4().hex[:8]
            path = PHOTOS / f"{iid}{pathlib.Path(f.name).suffix.lower()}"
            path.write_bytes(f.getvalue())
            try:
                item = clf.classify(str(path), item_id=iid)
            except Exception as e:      # model download or decode failure
                st.error(f"{f.name}: recognition failed ({e}). Added for you to label by hand.")
                try:
                    item = ItemClassifier(backend=None).classify(str(path), item_id=iid)
                except Exception:
                    path.unlink(missing_ok=True)
                    st.error(f"{f.name} could not be read as an image and was skipped.")
                    continue
            s.add_item(item)
            bar.progress((n + 1) / len(files), text=f"{n + 1} of {len(files)}")
        persist()
        clear_results()
        st.session_state.up_n += 1
        st.rerun()

    st.subheader("Your wardrobe")
    if not items:
        st.info("Nothing here yet. Upload a few tops, bottoms and shoes to get started.")
    only = st.toggle("Show only items to confirm", value=False) if items else False
    shown = [i for i in items if i.needs_review] if only else items
    cols = st.columns(4)
    for n, it in enumerate(shown):
        with cols[n % 4], st.container(border=True):
            show_photo(it)
            st.markdown(f"**{label(it)}**")
            if it.needs_review:
                guesses = ", ".join(f"{c} {p:.0%}" for c, p in it.alternatives)
                st.warning("Please confirm what this is." + (f" Best guesses: {guesses}" if guesses else ""))
            else:
                st.caption(f"{it.pattern} · confidence {it.confidence:.0%} · saved in {it.wear_count} outfit(s)")
            with st.expander("Correct or delete", expanded=it.needs_review):
                cats = list(CATEGORIES)
                cat = st.selectbox("Category", cats, key=f"cat{it.id}",
                                   index=cats.index(it.category) if it.category in cats else 0)
                colors = st.multiselect("Colours, main one first", list(COLOR_RGB),
                                        default=[c.name for c in it.colors], max_selections=3,
                                        key=f"col{it.id}")
                pats = PATTERNS + ([it.pattern] if it.pattern not in PATTERNS else [])
                pat = st.selectbox("Pattern", pats, index=pats.index(it.pattern), key=f"pat{it.id}")
                a, b = st.columns(2)
                if a.button("Save", key=f"save{it.id}", type="primary"):
                    s.correct_item(it.id, cat, colors or None, pat)
                    persist()
                    clear_results()
                    st.rerun()
                if b.button("Delete", key=f"del{it.id}"):
                    if it.image_path:
                        pathlib.Path(it.image_path).unlink(missing_ok=True)
                    s.remove_item(it.id)
                    persist()
                    clear_results()
                    st.rerun()

with t_prof:
    st.subheader("My preferences")
    st.caption("All optional. You can get outfits without filling any of this in.")
    p = s.profile
    styles = st.multiselect("Styles I like", STYLES, default=p.preferred_styles)
    a, b = st.columns(2)
    liked = a.multiselect("Colours I like", list(COLOR_RGB), default=p.liked_colors)
    avoided = b.multiselect("Colours I avoid", list(COLOR_RGB), default=p.avoided_colors)
    occs = st.multiselect("Occasions I dress for", [o.value for o in Occasion],
                          default=[Occasion(o).value for o in p.occasions])
    modest = st.checkbox("I prefer more coverage", value=p.min_coverage is not None)
    cov = st.slider("Minimum coverage (0.5 = short sleeves and knee length, 0.9 = long sleeves and full length)",
                    0.3, 1.0, p.min_coverage or 0.7, 0.05, disabled=not modest)
    always = st.multiselect("Always include, when I own one", ["hijab", "scarf", "watch", "belt", "cap"],
                            default=p.always_include)
    if st.button("Save preferences", type="primary"):
        p.preferred_styles, p.liked_colors, p.avoided_colors = styles, liked, avoided
        p.occasions = [Occasion(o) for o in occs]
        p.min_coverage = cov if modest else None
        p.always_include = always
        persist()
        clear_results()
        st.success("Saved.")

with t_out:
    st.subheader("What should I wear?")
    ctx = context_inputs("out")
    k = st.slider("How many suggestions", 1, 5, 3)
    if st.button("Suggest outfits", type="primary", key="go_out"):
        st.session_state.recs["out"] = (s.recommend(ctx, k), ctx)
    render_results("out")

with t_chal:
    st.subheader("Style challenge")
    st.caption("New looks built around a piece you rarely wear, skipping outfits you already saved.")
    owned = sorted({i.category for i in items if i.slot is not None})
    a, b = st.columns(2)
    cat = a.selectbox("Build around", ["My least-worn item"] + owned)
    pool = [i for i in items if i.slot is not None and (cat == "My least-worn item" or i.category == cat)]
    pick = b.selectbox("Or a specific piece", ["Choose for me"] + [i.id for i in pool],
                       format_func=lambda i: i if i == "Choose for me" else label(s.wardrobe[i]))
    cctx = context_inputs("chal")
    if st.button("Challenge me", type="primary", disabled=not pool):
        rec = s.challenge(None if cat == "My least-worn item" else cat,
                          None if pick == "Choose for me" else pick, cctx)
        st.session_state.recs["chal"] = (rec, cctx)
    render_results("chal")

with t_ins:
    a, b = st.columns(2)
    with a:
        st.subheader("Wardrobe gaps")
        st.caption("Based on the occasions in your preferences. These are kinds of garment, not products.")
        gaps = s.gaps() if items else []
        if not items:
            st.info("Add some items first.")
        elif not gaps:
            st.success("No gaps found for the occasions you dress for.")
        for g in gaps:
            (st.error if g.priority == "essential" else st.info)(
                f"{'Essential' if g.priority == 'essential' else 'Optional'}: {g.message}")
    with b:
        st.subheader("Your style direction")
        sugg = s.style_identity()
        if not sugg:
            st.info("Not enough to go on yet.")
        for sg in sugg:
            st.progress(sg.share, text=f"{sg.style}: {sg.share:.0%}")
            say(sg.evidence)
        if sugg and sugg[0].tentative:
            st.caption("Early guess: add more items or rate more outfits for a steadier picture.")
        if sugg:
            st.caption("These are suggestions. Change them any time under My preferences.")

with t_saved:
    st.subheader("Saved outfits")
    if not s.saved:
        st.info("Outfits you like will appear here.")
    for n, ids in enumerate(list(s.saved)):
        its = [s.wardrobe[i] for i in ids if i in s.wardrobe]
        with st.container(border=True):
            cols = st.columns(max(len(its), 5) + 1)
            for col, it in zip(cols, its):
                with col:
                    show_photo(it)
                    st.caption(label(it))
            if cols[-1].button("Remove", key=f"unsave{n}"):
                s.saved.pop(n)
                persist()
                st.rerun()
