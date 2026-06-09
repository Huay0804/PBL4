"""Overlay rendering for the Tooth Charting Assistant.

All renders are produced at the uploaded image's own resolution so the UI
preserves the original aspect ratio. Colors come from the stable palette in
config so a given tooth id always maps to the same color.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import PALETTE, NUM_CLASSES
from reporting import tooth_meta


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Map a (H, W) class-id array to an (H, W, 3) RGB image via PALETTE."""
    lut = np.array(PALETTE, dtype=np.uint8)            # (NUM_CLASSES, 3)
    safe = np.clip(label_map, 0, NUM_CLASSES - 1)
    return lut[safe]


def _blend(base_rgb: np.ndarray, color_rgb: np.ndarray, mask: np.ndarray,
           opacity: float) -> np.ndarray:
    out = base_rgb.astype(np.float32).copy()
    m = mask[..., None].astype(np.float32) * float(opacity)
    out = out * (1.0 - m) + color_rgb.astype(np.float32) * m
    return out.clip(0, 255).astype(np.uint8)


def semantic_overlay(base_rgb: np.ndarray, label_map: np.ndarray,
                     opacity: float = 0.45,
                     color: tuple[int, int, int] = (0, 220, 120)) -> np.ndarray:
    """Single-color highlight of all teeth (segmentation extent vs background)."""
    mask = label_map > 0
    color_img = np.empty_like(base_rgb)
    color_img[:] = color
    return _blend(base_rgb, color_img, mask, opacity)


def per_tooth_overlay(base_rgb: np.ndarray, label_map: np.ndarray,
                      opacity: float = 0.45, draw_labels: bool = True) -> np.ndarray:
    """Each tooth class painted its own palette color, with FDI text labels."""
    colored = colorize_label_map(label_map)
    mask = label_map > 0
    out = _blend(base_rgb, colored, mask, opacity)
    if draw_labels:
        out = _draw_labels(out, label_map)
    return out


def _draw_labels(image: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    out = image.copy()
    h = out.shape[0]
    scale = max(0.4, min(1.0, h / 1000.0))
    thickness = max(1, int(round(scale * 2)))
    for tooth_id in [int(v) for v in np.unique(label_map) if int(v) != 0]:
        ys, xs = np.where(label_map == tooth_id)
        if xs.size == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        meta = tooth_meta(tooth_id)
        text = str(meta["fdi"]) if meta["fdi"] is not None else str(tooth_id)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        org = (max(0, cx - tw // 2), min(h - 1, cy + th // 2))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def semantic_mask_png_array(label_map: np.ndarray) -> np.ndarray:
    """Single-channel uint8 class-id mask (0..32), matching the dataset format."""
    return np.clip(label_map, 0, 255).astype(np.uint8)
