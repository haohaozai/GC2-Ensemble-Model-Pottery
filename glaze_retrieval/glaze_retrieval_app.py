"""Compact GC2 glaze search interface, local MPS and hosted CUDA/CPU."""
from __future__ import annotations
import hashlib
import html
import importlib.util
import os
import json
import re
import time
from threading import Lock
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

from glazeprep import config as C
from model_status import statuses
import pandas as pd

ROOT = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT / "sample_images"
DESCRIPTION_FILE = ROOT / "demo_content.json"

def demo_content():
    try:
        return json.loads(DESCRIPTION_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

CONTENT = demo_content()
# Limit fonts to common browser-safe choices; do not inject arbitrary JSON into CSS.
FONT_OPTIONS = {
    "System default": 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    "Arial": 'Arial, Helvetica, sans-serif',
    "Helvetica": 'Helvetica, Arial, sans-serif',
    "Georgia": 'Georgia, serif',
    "Verdana": 'Verdana, Geneva, sans-serif',
    "Trebuchet MS": '"Trebuchet MS", Arial, sans-serif',
    "Times New Roman": '"Times New Roman", Times, serif',
    "Courier New": '"Courier New", Courier, monospace',
}
APP_TITLE = str(CONTENT.get("title") or "GC2 Glaze Finder")
APP_FONT = FONT_OPTIONS.get(str(CONTENT.get("font", "System default")), FONT_OPTIONS["System default"])
BUTTON_COLOR = str(CONTENT.get("button_color", "#2E6D64"))
if not re.fullmatch(r"#[0-9a-fA-F]{6}", BUTTON_COLOR):
    BUTTON_COLOR = "#2E6D64"

# Configure which catalog fields appear below each recipe image.
# Each item is {"field": "CSV column name", "label": "Display label"}.
DEFAULT_RECIPE_FIELDS = [
    {"field": "recipe_id", "label": "Recipe ID"},
    {"field": "Cone", "label": "Cone"},
    {"field": "Surface", "label": "Surface"},
]
raw_fields = CONTENT.get("recipe_fields", DEFAULT_RECIPE_FIELDS)
RECIPE_FIELDS = [
    {"field": item["field"], "label": str(item.get("label", item["field"]))}
    for item in raw_fields
    if isinstance(item, dict) and isinstance(item.get("field"), str) and item["field"]
][:12] if isinstance(raw_fields, list) else DEFAULT_RECIPE_FIELDS
RECIPE_TITLE_FIELD = str(CONTENT.get("recipe_title_field", "Name"))
DIVERSITY_CAPTION = str(CONTENT.get("diversity_caption", "Broader variety: at most 2 recipes per cluster."))
DIVERSITY_HELP = str(CONTENT.get("diversity_help", "Limits each K-means cluster to two recipes where possible, showing a wider variety of glaze groups. Ranking scores are unchanged."))

def display_value(value):
    """Skip absent/NaN fields without showing 'nan' in the UI."""
    if value is None:
        return None
    try:
        import pandas as pd
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    value = str(value).strip()
    return value if value else None


SAMPLES = sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}) if SAMPLES_DIR.exists() else []

st.set_page_config(page_title=APP_TITLE, page_icon="🎨", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
/* Keep the app clear of Streamlit's fixed top toolbar/header. */
[data-testid="stHeader"] {height:2.65rem; background:transparent}
.block-container {padding-top:3.8rem!important;padding-bottom:2rem!important;max-width:1450px}
h1 {font-size:1.45rem!important;margin:0 0 .65rem 0!important;padding:0!important}
[data-testid="stHeadingWithActionElements"]:has(h1) {margin-bottom:.5rem!important}
h3 {font-size:.96rem!important;margin:.12rem 0!important}
[data-testid="stVerticalBlock"] {gap:.28rem}
[data-testid="stHorizontalBlock"] {gap:.55rem}
[data-testid="stImage"] img {max-height:170px;object-fit:contain}
[data-testid="stFileUploader"] section {min-height:43px;padding:.2rem .4rem}
[data-testid="stFileUploader"] section button {padding:.1rem .4rem}
/* Keep native Streamlit icon fonts intact (Material ligatures otherwise show as text). */
.material-icons, .material-symbols-rounded, .material-symbols-outlined,
[data-testid="stExpanderToggleIcon"], [data-testid="stFileUploader"] [data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded", "Material Icons" !important;
  font-weight: normal !important;
  font-style: normal !important;
  letter-spacing: normal !important;
  line-height: 1 !important;
  white-space: nowrap !important;
  -webkit-font-feature-settings: "liga" !important;
  font-feature-settings: "liga" !important;
}
/* Remove uploader decorative icon, not the file controls. */
[data-testid="stFileUploader"] [data-testid="stIconMaterial"] {display:none!important}
[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"] {min-width:0}

[data-testid="stAlert"] {padding:.3rem .5rem}
[data-testid="stMetric"] {padding:0}
.recipe-title {font-weight:650;font-size:.92rem;line-height:1.35;min-height:2.65rem;overflow-wrap:anywhere;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;margin-top:.95rem;margin-bottom:.35rem}
.small {font-size:.74rem;opacity:.78}
.swatch {display:inline-block;width:16px;height:16px;border:1px solid #8888;border-radius:4px;vertical-align:middle;margin-right:4px}
/* Fixed-size thumbnail frames make every result card line up. */
.recipe-photo {height:148px;display:flex;align-items:center;justify-content:center;overflow:hidden;border-radius:7px;background:rgba(127,127,127,.08)}
.recipe-photo img {width:100%;height:100%;object-fit:contain}
.recipe-empty {font-size:.7rem;opacity:.6}
.recipe-meta {font-size:.78rem;opacity:.8;min-height:2.5rem;line-height:1.5;overflow-wrap:anywhere;margin-bottom:.8rem}
.recipe-meta-row {margin:.12rem 0}
[data-testid="stButton"] button[kind="primary"], [data-testid="stDownloadButton"] button[kind="primary"] {background-color:__BUTTON_COLOR__!important;border-color:__BUTTON_COLOR__!important;color:white!important}
[data-testid="stButton"] button[kind="primary"]:hover {filter:brightness(.88)}
[data-testid="stLinkButton"] button {min-height:2.2rem}
/* A little breathing room between rows of cards. */
[data-testid="stVerticalBlock"] > div:has(.recipe-photo) {min-width:0}
/* Consistent font across native widgets, custom HTML recipe cards and links. */
[data-testid="stApp"] .recipe-title,
[data-testid="stApp"] .recipe-meta,
[data-testid="stApp"] .recipe-meta-row,
[data-testid="stApp"] .recipe-empty,
[data-testid="stApp"] .small,
[data-testid="stApp"] [data-testid="stLinkButton"] button,
[data-testid="stApp"] [data-testid="stLinkButton"] button *,
[data-testid="stApp"] [data-testid="stMarkdownContainer"] :is(p, strong, em, a, li),
[data-testid="stApp"] [data-testid="stSelectbox"] *,
[data-testid="stApp"] [data-testid="stToggle"] label {
  font-family: __APP_FONT__ !important;
}
</style>""".replace("__APP_FONT__", APP_FONT).replace("__BUTTON_COLOR__", BUTTON_COLOR) + f"<style>html, body, [data-testid='stApp'] {{font-family:{APP_FONT};}} [data-testid='stApp'] p, [data-testid='stApp'] h1, [data-testid='stApp'] h2, [data-testid='stApp'] h3, [data-testid='stApp'] h4, [data-testid='stApp'] label, [data-testid='stApp'] button, [data-testid='stApp'] input, [data-testid='stApp'] textarea {{font-family:{APP_FONT};}}</style>", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading retrieval encoder…")
def get_retriever():
    import retrieval_service
    return retrieval_service

@st.cache_resource
def get_palette_function():
    source = ROOT / "scripts" / "17_colorthief_colors.py"
    spec = importlib.util.spec_from_file_location("gc2_colorthief", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.colorthief_palette

@st.cache_resource(show_spinner="Loading Grounding DINO, SAM 2.1 and Ceramic CLIP…")
def get_segmenter():
    """Lazy singleton: initialize the three models on the first Detect click only.

    Streamlit retains this resource across widget reruns, sample changes and
    image uploads. It is rebuilt only if the server/cache is reset or this
    cached function's source changes.
    """
    from segmentation_frontend import load_segmenter
    return load_segmenter()


@st.cache_resource

def get_segmentation_lock():
    """Serialize access to the shared model on concurrent browser sessions."""
    return Lock()

st.title("🎨 " + APP_TITLE)
if CONTENT.get("intro"):
    # Markdown supports paragraphs, blank lines, bold, lists and links.
    # JSON may provide a single string (with \n) or a list of paragraphs.
    intro = CONTENT["intro"]
    if isinstance(intro, list):
        intro = "\n\n".join(str(part) for part in intro)
    if isinstance(intro, str):
        st.markdown(intro)
model_ready = "Ready (reused for subsequent images)" if "segmenter_load_seconds" in st.session_state else "Not loaded (loads on first detection)"
timing = f" · Last segmentation {st.session_state['segmentation_seconds']:.1f}s" if "segmentation_seconds" in st.session_state else ""
st.caption(f"Inference: **{C.DEVICE.upper()}** · Segmentation models: **{model_ready}**{timing}")
# Full-width expander avoids Streamlit's popover icon overlapping a long button label.
with st.expander("Model availability / diagnostics", expanded=False):
    st.write(f"Device: `{C.DEVICE}` · MPS: `{torch.backends.mps.is_available()}` · CPU fallback: `{bool(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'))}`")
    if "segmenter_load_seconds" in st.session_state:
        st.success(f"First model initialization: {st.session_state['segmenter_load_seconds']:.1f}s. Subsequent images reuse the loaded models.")
    for name, ready in statuses().items():
        if ready is True:
            st.success(f"{name}: cached weights detected")
        elif ready is False:
            st.warning(f"{name}: weights not cached. First use downloads them (internet required).")
        else:
            st.info(f"{name}: cache not confirmed; verified at model load.")
    st.caption("Checks are best-effort. Hosted machines may lose caches after restart.")

upload_col, object_col, settings_col = st.columns([1.05, 1.05, 1.45], gap="medium")
with upload_col:
    st.markdown("### 1 · Photo")
    source = st.radio("Image source", ["Upload", "Sample image"], horizontal=True, label_visibility="collapsed") if SAMPLES else "Upload"
    uploaded = None
    photo = None
    photo_bytes = None
    if source == "Sample image":
        sample = st.selectbox("Choose sample", SAMPLES, format_func=lambda p: p.stem.replace("_", " ").replace("-", " "), label_visibility="collapsed")
        photo_bytes = sample.read_bytes()
        description = CONTENT.get("samples", {}).get(sample.name, "")
        if description:
            st.caption(str(description))
    else:
        uploaded = st.file_uploader("Upload JPG / PNG / WebP", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
        if uploaded is not None:
            photo_bytes = uploaded.getvalue()
    if photo_bytes:
        try:
            import io
            photo = ImageOps.exif_transpose(Image.open(io.BytesIO(photo_bytes))).convert("RGB")
            st.image(photo, width=185)
        except Exception:
            st.error("Cannot read this image.")
    detect = st.button("Detect pottery", type="primary", use_container_width=True, disabled=photo is None)

if photo is not None:
    photo_key = hashlib.sha256(photo_bytes).hexdigest()
    if st.session_state.get("segmentation_photo_key") != photo_key:
        for key in ("segmented_objects", "segmentation_meta", "results_regular", "results_diversified", "results_query", "segmentation_seconds", "palette"):
            st.session_state.pop(key, None)
        st.session_state["segmentation_photo_key"] = photo_key
    if detect:
        try:
            from segmentation_frontend import extract_objects
            with st.status("Loading models and segmenting…", expanded=True) as status:
                load_start = time.perf_counter()
                segmenter = get_segmenter()
                if "segmenter_load_seconds" not in st.session_state:
                    st.session_state["segmenter_load_seconds"] = time.perf_counter() - load_start
                st.write(f"Models ready on {segmenter.device}")
                start = time.perf_counter()
                # Cached models are shared between sessions; protect their use
                # against simultaneous inference from multiple browser tabs.
                with get_segmentation_lock():
                    objects, meta = extract_objects(photo, segmenter)
                st.session_state["segmentation_seconds"] = time.perf_counter() - start
                st.session_state["segmented_objects"] = objects
                st.session_state["segmentation_meta"] = meta
                st.session_state.pop("results_regular", None)
                st.session_state.pop("results_diversified", None)
                status.update(label=f"Detected {len(objects)} object(s) in {st.session_state['segmentation_seconds']:.1f}s", state="complete", expanded=False)
        except Exception as exc:
            st.error(f"Segmentation failed: {type(exc).__name__}: {exc}")
            st.info("Check model access, available memory and device settings.")

objects = st.session_state.get("segmented_objects") if photo is not None else None
palette = None
object_choice = None
with object_col:
    st.markdown("### 2 · Segmented object")
    if objects:
        object_choice = st.selectbox("Detected object", range(len(objects)), format_func=lambda i: f"Object {objects[i]['index']}" + (" · primary" if objects[i]['primary'] else ""), label_visibility="collapsed")
        selected_object = objects[object_choice]
        st.image(selected_object["image"], width=185)
        try:
            palette, note = get_palette_function()(selected_object["image"])
            if palette:
                st.markdown(" ".join(f'<span title="{html.escape(str(v["name"]))} · {v["fraction"]:.0%}"><span class="swatch" style="background:{html.escape(str(v["hex"]))}"></span><span class="small">{html.escape(str(v["hex"]))}</span></span>' for v in palette[:5]), unsafe_allow_html=True)
            else:
                st.warning("No usable glaze colours found.")
        except Exception as exc:
            st.error(f"Colour extraction failed: {exc}")
    elif objects == []:
        st.warning("No ceramic objects detected. Try a clearer photo.")
    else:
        st.caption("Detect pottery to preview the isolated object and extracted colours.")

with settings_col:
    st.markdown("### 3 · Recipe search")
    chosen = st.selectbox("Glaze colour", range(len(palette)) if palette else [0], format_func=lambda i: f'{palette[i]["name"]} · {palette[i]["hex"]}' if palette else "Detect an object first", disabled=not palette)
    a, b = st.columns(2)
    with a:
        surface = st.selectbox("Surface", ["Glossy", "Matte", "Satin", "Semi-Matte", "Dry", "Unknown"])
    with b:
        cone = st.text_input("Firing cone", value="6")
    top_k = st.selectbox("Results", list(range(3, 9)), index=5)
    search = st.button("Find matching recipes", type="primary", use_container_width=True, disabled=not palette)


if search and palette and photo is not None:
    if not cone.strip():
        st.warning("Enter a firing cone.")
    else:
        try:
            with st.spinner("Retrieving standard and diversified results…"):
                selected = palette[chosen]
                retriever = get_retriever()
                params = dict(color=selected["name"], surface=surface, cone=cone,
                              query_hex=selected["hex"], top_k=top_k)
                regular = retriever.retrieve_glazes(**params, diversify=False)
                diversified = retriever.retrieve_glazes(**params, diversify=True, max_per_cluster=2)
            st.session_state["results_regular"] = regular
            st.session_state["results_diversified"] = diversified
            st.session_state["results_query"] = (photo_key, object_choice, chosen, surface, cone, top_k)
        except Exception as exc:
            st.error(f"Retrieval failed: {exc}")
            st.info("Check the pretrained MiniLM model and bundled retrieval assets.")

@st.fragment
def render_results():
    if palette and photo is not None and st.session_state.get("results_query") == (photo_key, object_choice, chosen, surface, cone, top_k):
        regular_results = st.session_state.get("results_regular")
        diversified_results = st.session_state.get("results_diversified")
        if regular_results is not None and diversified_results is not None and len(regular_results):
            st.divider()
            header, toggle_col, export = st.columns([2.5, 3, 1.2], vertical_alignment="center")
            with header:
                st.markdown("### 4 · Matching recipes")
            with toggle_col:
                diversify_view = st.toggle("Diversify clusters", key="diversify_results_view",
                    help=DIVERSITY_HELP)
                st.caption(DIVERSITY_CAPTION)
            # Display-only options: toggling these does not rerun retrieval.
            score_col1, score_col2, _ = st.columns([1.35, 1.35, 3.5], vertical_alignment="center")
            with score_col1:
                show_latent_score = st.checkbox(
                    "Show latent score", value=False, key="show_latent_score",
                    help="Normalized similarity from the learned latent embedding space (0–100%).",
                )
            with score_col2:
                show_color_score = st.checkbox(
                    "Show colour score", value=False, key="show_color_score",
                    help="CIEDE2000-based colour similarity (0–100%).",
                )
            results = diversified_results if diversify_view else regular_results
            with export:
                st.download_button("Export CSV", results.to_csv(index=False).encode("utf-8"),
                    file_name="glaze_matches_diversified.csv" if diversify_view else "glaze_matches_regular.csv",
                    mime="text/csv", use_container_width=True)
            # Four columns per row (up to two rows), keeping photos and links aligned.
            records = list(results.iterrows())
            for row_start in range(0, len(records), 4):
                cols = st.columns(4, gap="medium")
                for col, (_, rec) in zip(cols, records[row_start:row_start + 4]):
                    with col:
                        url = rec.get("image_path")
                        if isinstance(url, str) and url.startswith(("https://", "http://")):
                            safe_url = html.escape(url, quote=True)
                            st.markdown(f'<div class="recipe-photo"><img src="{safe_url}" alt="Recipe photo" loading="lazy"></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="recipe-photo"><span class="recipe-empty">No image</span></div>', unsafe_allow_html=True)
                        name = display_value(rec.get(RECIPE_TITLE_FIELD)) or f'Recipe {rec["recipe_id"]}'
                        st.markdown(f'<div class="recipe-title" title="{html.escape(str(name), quote=True)}">{html.escape(str(name))}</div>', unsafe_allow_html=True)
                        details = []
                        score = rec.get("final_score")
                        if score is not None:
                            try:
                                score_number = float(score)
                                if pd.notna(score_number):
                                    details.append(f'<div class="recipe-meta-row"><strong>Similarity score:</strong> {score_number * 100:.2f}%</div>')
                            except (TypeError, ValueError):
                                pass
                        for enabled, field_name, label in (
                            (show_latent_score, "latent_score", "Latent score"),
                            (show_color_score, "color_score", "Colour score"),
                        ):
                            if enabled:
                                try:
                                    value = float(rec.get(field_name))
                                    if np.isfinite(value):
                                        details.append(
                                            f'<div class="recipe-meta-row"><strong>{label}:</strong> {value * 100:.2f}%</div>'
                                        )
                                except (TypeError, ValueError):
                                    pass
                        if diversify_view:
                            cluster_value = rec.get("kmeans_cluster")
                            if cluster_value is not None and pd.notna(cluster_value):
                                cluster_label = html.escape(str(cluster_value))
                                details.append(
                                    f'<div class="recipe-meta-row"><strong>Cluster:</strong> {cluster_label}</div>'
                                )
                        for item in RECIPE_FIELDS:
                            value = display_value(rec.get(item["field"]))
                            if value is not None:
                                details.append(f'<div class="recipe-meta-row"><strong>{html.escape(item["label"])}:</strong> {html.escape(value)}</div>')
                        st.markdown('<div class="recipe-meta">' + ''.join(details) + '</div>', unsafe_allow_html=True)
                        st.link_button("View recipe", f'https://glazy.org/recipes/{rec["recipe_id"]}', use_container_width=True)
                st.write("")


render_results()
if photo is None:
    st.caption("Upload a photo to begin. Models are loaded only when needed.")
