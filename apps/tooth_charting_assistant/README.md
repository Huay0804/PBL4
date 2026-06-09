# 🦷 AI Dental Panoramic Tooth Charting Assistant

A lightweight Streamlit MVP that runs this project's trained segmentation models
on a panoramic dental X-ray and visualizes the predicted tooth
segmentation + numbering, with export to PNG / JSON / CSV.

> **Annotation / review aid only.** This is **not** a medical device and does
> **not** provide a diagnosis. Every output must be verified by a qualified
> clinician.

It is isolated from the research/training/evaluation pipeline: it does not modify
any existing scripts, and it reads model weights from its **own**
[`models/`](models/) directory rather than from the project's training output
tree. The default and recommended model is **YOLO11-seg** (instance-oriented,
fast, and competitive with Mod NestNet / ICPR MUNet in this project). YOLO26-seg
is available but underperformed, so it is not the default.

## Install & run

```bash
# from the repo root
python -m venv .venv-app && source .venv-app/bin/activate   # optional, keeps it isolated
pip install -r apps/tooth_charting_assistant/requirements.txt
streamlit run apps/tooth_charting_assistant/app.py
```

Then open the URL Streamlit prints (default http://localhost:8501), upload a
panoramic X-ray, and click **Run inference**.

## Where to put model weights

The app reads weights from its **own** model directory and does **not** touch the
PBL4 research/training output tree. By default that directory is:

```
apps/tooth_charting_assistant/models/
```

Override it by setting the `TCA_MODELS_DIR` environment variable to any folder.

Drop checkpoints in named after the model's `run_name` (first match wins):

```
models/yolo11_seg.pt              # default model
models/yolo26_seg.pt              # optional
models/yolo11_seg_cv_summary.json # optional — shows CV metrics in the sidebar
```

A training-style subfolder also resolves
(`models/yolo11_seg/weights/best.pt`). See [models/README.md](models/README.md)
for the full naming rules. To seed the directory from this project's CV runs:

```bash
cp yolo_seg_models/runs/cv/fold_0/yolo11_seg/weights/best.pt \
   apps/tooth_charting_assistant/models/yolo11_seg.pt
```

or download `yolo_seg_models.zip` from the Kaggle notebook and copy the desired
`best.pt` into `models/`.

If **no checkpoint is found** the app still runs — it shows a clear
"checkpoint not found" state and lets you paste a custom `.pt` path in the
sidebar. Nothing fails at startup if weights are missing.

## Models available

| Sidebar option | Backend | Notes |
|---|---|---|
| **YOLO11-seg (recommended)** | Ultralytics `.pt` | Default. Instance masks → 33-class semantic map. |
| YOLO26-seg | Ultralytics `.pt` | Optional; underperformed in this project. |
| Custom YOLO checkpoint (.pt) | Ultralytics `.pt` | Paste any YOLO-seg checkpoint path. |
| TransUNet (dense, image-only) | Keras `.keras` | Optional. Needs TensorFlow installed. |
| ICPR MUNet / Mod NestNet (dense) | Keras `.keras` | These are **BB-gated** (need bounding-box prior maps from the YOLOX / Mask R-CNN pipeline), which is out of scope for this MVP. The app loads them but reports clearly that priors are required. |

Dense (Keras) models are optional and require a working TensorFlow/Keras
environment; they are intentionally not in `requirements.txt` to keep the app
lightweight. Use a YOLO model if you don't have TensorFlow set up.

## What it shows

- **Per-tooth overlay** — each tooth class painted with a stable color, labeled
  with its derived FDI number.
- **Semantic overlay** — single-color highlight of the segmentation extent.
- **Original** image (aspect ratio preserved throughout).
- **Detected teeth table** — `tooth_id`, derived `fdi`, `class_name`,
  `quadrant`, `tooth_type`, `confidence` (YOLO only), `mask_area_pixels`,
  `bbox_xyxy`.
- **Summary** — teeth detected, mean confidence, tooth-area fraction.

## Image handling & performance

- **Input images** are normalized on upload: EXIF orientation is honored,
  16-bit / float grayscale dental scans are min-max scaled to 8-bit (instead of
  being clipped), and images above ~40 MP are rejected with a clear message
  rather than exhausting memory. Aspect ratio is always preserved.
- **Weights load once.** The selected checkpoint is loaded into memory the first
  time you run inference and cached for the session (`st.cache_resource`), so
  repeat runs don't re-read the `.pt` from disk.
- **Cheap re-renders.** The prediction is cached per (image, model, confidence,
  IoU). Moving the **overlay opacity** slider re-renders the overlays instantly
  from the cached masks — it does *not* re-run the model. Changing the
  confidence / IoU threshold or the image prompts you to click **Run inference**
  again, since those change the prediction itself.

## Exports

- **Overlay PNG** — the colored per-tooth overlay.
- **Semantic mask PNG** — single-channel class-id mask (`0` = background,
  `1..32` = teeth), matching the dataset's `masks_semantic` format.
- **JSON report** — model/params, image size, summary, and the full per-tooth list.
- **CSV table** — the tooth table.

## Conventions (consistent with the research code)

- 33 classes: `0` = background, `1..32` = teeth. YOLO's 0-based class `c` maps to
  tooth id `c + 1`.
- Instance masks are collapsed into one semantic label map by painting masks in
  ascending-confidence order (the most confident tooth wins overlaps) — the same
  rule as `scripts/yolo_seg_utils.rasterize_result`.
- FDI numbers and quadrant / position / tooth-type labels are derived from the
  quadrant map in `scripts/segmentation_report.py`.
- Dense models run at the protocol resolution 512×1024; overlays are rendered at
  the uploaded image's native resolution.

## Files

```
apps/tooth_charting_assistant/
  app.py            Streamlit UI (sidebar, tabs, table, exports)
  config.py         constants, palette, model registry, checkpoint discovery
  inference.py      YOLO + dense backends -> unified Prediction
  visualization.py  semantic + per-tooth overlays
  reporting.py      tooth table, JSON report, CSV, FDI/quadrant labels
  requirements.txt
  README.md
```
