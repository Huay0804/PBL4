"""AI Dental Panoramic Tooth Charting Assistant — Streamlit MVP.

Run:
    streamlit run apps/tooth_charting_assistant/app.py

Uploads a panoramic dental X-ray, runs a selected segmentation model (YOLO11-seg
by default), and shows overlays + a per-tooth table with export options.

This is an annotation / review aid, NOT a diagnostic device.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

import config
from inference import BackendError, build_backend
import reporting
import visualization as viz


st.set_page_config(page_title="Tooth Charting Assistant", page_icon="🦷", layout="wide")

# Guard against pathological uploads that would blow up memory on resize/overlay.
MAX_PIXELS = 40_000_000  # ~40 megapixels


def _png_bytes(rgb_array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="PNG")
    return buf.getvalue()


def _cv_summary_text(run_name: str | None) -> str | None:
    path = config.find_cv_summary(run_name)
    if not path:
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    agg = data.get("aggregate", {})
    miou = agg.get("macro_iou", {}).get("mean")
    mdice = agg.get("macro_dice", {}).get("mean")
    if miou is None:
        return None
    return f"CV macro IoU {miou:.3f} · macro Dice {mdice:.3f} ({data.get('folds','?')} folds)"


def load_image(uploaded) -> Image.Image:
    """Read an uploaded X-ray into an RGB PIL image, robust to real-world quirks.

    Honors EXIF orientation, scales high-bit-depth (16-bit / float) grayscale
    scans to 8-bit by min-max instead of clipping, and rejects images above a
    sane pixel budget so the UI fails with a clear message rather than OOM-ing.
    """
    img = Image.open(uploaded)
    img = ImageOps.exif_transpose(img)  # rotate per camera/scanner metadata
    if img.height * img.width > MAX_PIXELS:
        mp = img.height * img.width / 1e6
        raise ValueError(
            f"Image is {img.width}×{img.height} ({mp:.1f} MP), above the "
            f"{MAX_PIXELS / 1e6:.0f} MP limit. Please downscale and retry."
        )
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(img).astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))  # 2-D uint8 -> mode "L"
    return img.convert("RGB")


@st.cache_resource(show_spinner=False)
def _load_backend(backend_type: str, run_name, ckpt_path, model_name: str):
    """Cached backend (and its loaded weights) keyed by hashable params.

    Streamlit reruns the whole script on every interaction; caching here means
    the heavy checkpoint is loaded from disk once per (model, checkpoint) and
    reused for subsequent inferences in the session.
    """
    model_def = {"backend": backend_type, "run_name": run_name}
    return build_backend(model_def, ckpt_path, model_name)


def run_inference(model_def, ckpt_path, model_name, image, conf, iou):
    backend = _load_backend(
        model_def["backend"], model_def.get("run_name"), ckpt_path, model_name
    )
    return backend.predict(image, conf=conf, iou=iou)


def sidebar():
    st.sidebar.header("Model & inference")
    model_name = st.sidebar.selectbox(
        "Model", list(config.MODELS.keys()),
        index=list(config.MODELS.keys()).index(config.DEFAULT_MODEL),
    )
    model_def = config.MODELS[model_name]

    auto_ckpt = config.find_checkpoint(model_def)
    is_custom = model_def.get("custom", False)

    if is_custom:
        st.sidebar.caption("Provide a path to a YOLO-seg `.pt` checkpoint.")
        ckpt_input = st.sidebar.text_input("Checkpoint path (.pt)", value="")
    else:
        default_val = str(auto_ckpt) if auto_ckpt else ""
        ckpt_input = st.sidebar.text_input(
            "Checkpoint path (auto-detected, editable)", value=default_val,
            help="Leave as detected, or point at your own checkpoint.",
        )

    ckpt_path = ckpt_input.strip() or None
    if ckpt_path:
        if Path(ckpt_path).exists():
            st.sidebar.success(f"Checkpoint: {ckpt_path}")
        else:
            st.sidebar.error(f"File not found: {ckpt_path}")
            ckpt_path = None  # treat a bad path as "no checkpoint", never crash
    else:
        st.sidebar.warning(
            "Checkpoint not found. See the README for where to place weights, "
            "or paste a path above. You can still explore the UI."
        )

    summary = _cv_summary_text(model_def.get("run_name"))
    if summary:
        st.sidebar.caption(f"📊 {summary}")

    st.sidebar.divider()
    conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, config.DEFAULT_CONF, 0.01)
    iou = st.sidebar.slider("IoU (NMS) threshold", 0.0, 1.0, config.DEFAULT_IOU, 0.01)
    opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, config.DEFAULT_OPACITY, 0.05)
    if model_def["backend"] == "dense":
        st.sidebar.info("Dense models are semantic-only: confidence/IoU are ignored.")
    return model_name, model_def, ckpt_path, conf, iou, opacity


def render_results(pred, base_rgb, image, opacity):
    """Render metrics, overlays, table and exports for a computed prediction.

    Runs on every rerun (it is cheap), so moving the opacity slider re-renders
    overlays instantly without re-running inference.
    """
    df = reporting.build_table(pred)
    report = reporting.build_report(pred, df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Teeth detected", pred.num_teeth)
    mean_conf = report["summary"]["mean_confidence"]
    c2.metric("Mean confidence", f"{mean_conf:.3f}" if mean_conf is not None else "n/a")
    frac = report["summary"]["tooth_area_fraction"]
    c3.metric("Tooth area fraction", f"{frac:.1%}" if frac is not None else "n/a")

    per_tooth_img = viz.per_tooth_overlay(base_rgb, pred.label_map, opacity)
    semantic_img = viz.semantic_overlay(base_rgb, pred.label_map, opacity)

    tab_pt, tab_sem, tab_orig = st.tabs(
        ["Per-tooth overlay", "Semantic overlay", "Original"]
    )
    with tab_pt:
        st.image(per_tooth_img, use_container_width=True,
                 caption="Each tooth colored by class; numbers are derived FDI labels.")
    with tab_sem:
        st.image(semantic_img, use_container_width=True,
                 caption="Segmentation extent (teeth vs background).")
    with tab_orig:
        st.image(image, use_container_width=True)

    st.subheader("Detected teeth")
    if df.empty:
        st.warning("No teeth detected. Try lowering the confidence threshold.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Export")
    e1, e2, e3, e4 = st.columns(4)
    e1.download_button(
        "Overlay PNG", _png_bytes(per_tooth_img),
        file_name="tooth_overlay.png", mime="image/png",
    )
    e2.download_button(
        "Semantic mask PNG", _png_bytes(viz.semantic_mask_png_array(pred.label_map)),
        file_name="semantic_mask.png", mime="image/png",
        help="Single-channel class-id mask (0=background, 1..32=teeth).",
    )
    e3.download_button(
        "JSON report", reporting.report_json_bytes(report),
        file_name="tooth_report.json", mime="application/json",
    )
    e4.download_button(
        "CSV table", reporting.table_csv_bytes(df),
        file_name="tooth_table.csv", mime="text/csv",
    )


def main():
    st.title("🦷 AI Dental Panoramic Tooth Charting Assistant")
    st.caption(config.DISCLAIMER)

    model_name, model_def, ckpt_path, conf, iou, opacity = sidebar()

    uploaded = st.file_uploader(
        "Upload a panoramic dental X-ray", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )
    if uploaded is None:
        st.session_state.pop("pred", None)
        st.session_state.pop("pred_key", None)
        st.info("Upload a panoramic X-ray to begin.")
        return

    try:
        image = load_image(uploaded)
    except Exception as exc:
        st.error(f"Could not read image: {exc}")
        return
    base_rgb = np.asarray(image)

    # Cache key for the prediction. Opacity is deliberately excluded so changing
    # it re-renders overlays from the cached prediction without re-inferring;
    # conf/iou/model/image are included because they change the prediction.
    key = (
        hashlib.md5(image.tobytes()).hexdigest(),
        model_name, ckpt_path or "", round(conf, 4), round(iou, 4),
    )

    run = st.button("Run inference", type="primary", disabled=ckpt_path is None)
    if ckpt_path is None:
        st.warning("No checkpoint available — provide one in the sidebar to run inference.")

    if run:
        with st.spinner(f"Running {model_name}…"):
            try:
                pred = run_inference(model_def, ckpt_path, model_name, image, conf, iou)
            except BackendError as exc:
                st.error(str(exc))
                return
            except Exception as exc:  # pragma: no cover - surface unexpected errors cleanly
                st.error(f"Inference failed: {exc}")
                return
        st.session_state["pred"] = pred
        st.session_state["pred_key"] = key

    pred = st.session_state.get("pred")
    has_valid_pred = pred is not None and st.session_state.get("pred_key") == key

    if not has_valid_pred:
        st.subheader("Original image")
        st.image(image, use_container_width=True)
        if pred is not None:
            st.info("Inference settings changed. Click **Run inference** to refresh results.")
        return

    render_results(pred, base_rgb, image, opacity)
    st.caption(config.DISCLAIMER)


if __name__ == "__main__":
    main()
