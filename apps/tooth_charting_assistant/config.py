"""Static configuration for the Tooth Charting Assistant.

Self-contained: this module does NOT import the research pipeline. It only
mirrors a few stable conventions from it (33-class layout, the quadrant/position
map in scripts/segmentation_report.py) so the app stays isolated and runnable on
its own.
"""

from __future__ import annotations

import colorsys
import glob
import os
from pathlib import Path

# --- Project geometry / class convention ------------------------------------
# class 0 = background, classes 1..32 = teeth (matches data/splits masks).
NUM_CLASSES = 33
NUM_TEETH = 32
# Protocol working resolution (height, width). Used for the YOLO inference
# imgsz long side and noted in the report; overlays render at the uploaded
# image's own resolution to preserve aspect ratio.
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024
YOLO_IMGSZ = 1024

# --- Where the app looks for model weights ----------------------------------
# The app is self-contained: it reads checkpoints (and optional metrics) from
# its OWN models directory, NOT from the PBL4 research/training output tree.
# Override the location with the TCA_MODELS_DIR environment variable.
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("TCA_MODELS_DIR", APP_DIR / "models")).expanduser()

# REPO_ROOT is retained ONLY so the optional dense (Keras) backend can import the
# project's custom layers to deserialize a .keras model. The default YOLO path
# does not use it and is fully independent of the repo layout.
REPO_ROOT = Path(__file__).resolve().parents[2]


# --- Model registry ----------------------------------------------------------
# backend "yolo"  -> Ultralytics .pt, instance segmentation (default path).
# backend "dense" -> Keras .keras semantic segmenter (optional, needs TF).
MODELS: dict[str, dict] = {
    "YOLO11-seg (recommended)": {"backend": "yolo", "run_name": "yolo11_seg"},
    "YOLO26-seg": {"backend": "yolo", "run_name": "yolo26_seg"},
    "Custom YOLO checkpoint (.pt)": {"backend": "yolo", "run_name": None, "custom": True},
    "TransUNet (dense, image-only)": {"backend": "dense", "run_name": "transunet"},
    "ICPR MUNet (dense, needs priors)": {"backend": "dense", "run_name": "icpr_munet"},
    "Mod NestNet (dense, needs priors)": {"backend": "dense", "run_name": "mod_nestnet"},
}
DEFAULT_MODEL = "YOLO11-seg (recommended)"

# Checkpoint name patterns searched inside MODELS_DIR (first sorted match wins).
# Drop a weight in as e.g. models/yolo11_seg.pt, or keep a training-style
# subfolder (models/yolo11_seg/weights/best.pt) — both resolve.
YOLO_CKPT_GLOBS = [
    "{run}.pt",
    "{run}/best.pt",
    "{run}/weights/best.pt",
    "*{run}*.pt",
]
DENSE_CKPT_GLOBS = [
    "{run}.keras",
    "{run}/best.keras",
    "{run}/**/best.keras",
    "*{run}*.keras",
]

# CV summary files (optional, shown in the sidebar if present), also in MODELS_DIR.
CV_SUMMARY_GLOBS = [
    "{run}_cv_summary.json",
    "*{run}*cv_summary.json",
]

# --- Inference defaults ------------------------------------------------------
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.70
DEFAULT_OPACITY = 0.45

DISCLAIMER = (
    "This tool is an annotation / review aid for visualizing automatic tooth "
    "segmentation and numbering. It is **not** a medical device and does **not** "
    "provide a diagnosis. All outputs must be verified by a qualified clinician."
)


def find_checkpoint(model_def: dict) -> Path | None:
    """Return the first checkpoint for a model def inside MODELS_DIR, or None.

    Searches only the app-local models directory (or TCA_MODELS_DIR). Never
    raises — a missing checkpoint is a normal, handled state.
    """
    run = model_def.get("run_name")
    if not run:
        return None
    globs = YOLO_CKPT_GLOBS if model_def["backend"] == "yolo" else DENSE_CKPT_GLOBS
    for pattern in globs:
        matches = sorted(glob.glob(str(MODELS_DIR / pattern.format(run=run)), recursive=True))
        matches = [m for m in matches if Path(m).is_file()]
        if matches:
            return Path(matches[0])
    return None


def find_cv_summary(run_name: str | None) -> Path | None:
    if not run_name:
        return None
    for pattern in CV_SUMMARY_GLOBS:
        matches = sorted(glob.glob(str(MODELS_DIR / pattern.format(run=run_name))))
        matches = [m for m in matches if Path(m).is_file()]
        if matches:
            return Path(matches[0])
    return None


# --- Color palette -----------------------------------------------------------
def build_palette(num_classes: int = NUM_CLASSES) -> list[tuple[int, int, int]]:
    """Stable RGB palette: index 0 = background (black), 1..N-1 = distinct hues.

    Deterministic (no RNG) so a given tooth id always gets the same color.
    """
    palette = [(0, 0, 0)]
    teeth = max(num_classes - 1, 1)
    for i in range(teeth):
        # Spread hue around the wheel; alternate value/saturation for contrast
        # between neighbouring ids.
        hue = (i * 0.61803398875) % 1.0  # golden-ratio spacing
        sat = 0.65 if i % 2 == 0 else 0.90
        val = 0.95 if i % 3 else 0.75
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette.append((int(r * 255), int(g * 255), int(b * 255)))
    return palette


PALETTE = build_palette()
