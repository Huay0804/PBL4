# Model weights directory

This is the app's **own** model store. The Tooth Charting Assistant reads
checkpoints from here only — it does **not** reach into the PBL4 research /
training output tree. To use a different location, set the `TCA_MODELS_DIR`
environment variable to point elsewhere.

## Expected files

| File | Used by sidebar option |
|---|---|
| `yolo11_seg.pt` | **YOLO11-seg (recommended)** — the default model |
| `yolo26_seg.pt` | YOLO26-seg (optional) |
| `yolo11_seg_cv_summary.json` | optional — shows cross-validation metrics in the sidebar |
| `yolo26_seg_cv_summary.json` | optional |

Naming: a model whose sidebar `run_name` is `<run>` is found via any of
`<run>.pt`, `<run>/best.pt`, `<run>/weights/best.pt`, or `*<run>*.pt` (first
match wins). So you can drop a flat `yolo11_seg.pt` here, or copy a whole
training run folder (`yolo11_seg/weights/best.pt`) — both resolve.

For a **custom** checkpoint you don't want to place here, pick
"Custom YOLO checkpoint (.pt)" in the sidebar and paste an absolute path.

## How to populate it

These `.pt` files are git-ignored (weights are large), so they are not
committed. Copy a trained checkpoint in, e.g. from this project's CV runs:

```bash
cp <repo>/yolo_seg_models/runs/cv/fold_0/yolo11_seg/weights/best.pt \
   apps/tooth_charting_assistant/models/yolo11_seg.pt
```

or download `yolo_seg_models.zip` from the Kaggle notebook and copy the
`best.pt` you want here. If this directory is empty the app still launches and
shows a clear "checkpoint not found" state.
