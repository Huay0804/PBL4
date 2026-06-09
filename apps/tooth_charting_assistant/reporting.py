"""Tooth-level reporting: FDI/position labels, table, JSON and CSV export.

The quadrant / position / tooth-type mapping mirrors
scripts/segmentation_report.py so labels are consistent with the research
metrics. FDI numbers are *derived* from that mapping for readability and are
clearly marked as derived (the underlying model predicts class ids 1..32).
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import pandas as pd

from inference import Prediction

# --- Mirrors scripts/segmentation_report.py ---------------------------------
_QUADRANT_DEFINITIONS = (
    (1, 8, "upper_left", "molar_to_incisor"),
    (9, 16, "upper_right", "incisor_to_molar"),
    (17, 24, "lower_right", "molar_to_incisor"),
    (25, 32, "lower_left", "incisor_to_molar"),
)
_POSITION_LABELS = {
    1: "central_incisor",
    2: "lateral_incisor",
    3: "canine",
    4: "first_premolar",
    5: "second_premolar",
    6: "first_molar",
    7: "second_molar",
    8: "third_molar",
}
# Project quadrant name -> FDI quadrant digit.
_FDI_QUADRANT_DIGIT = {
    "upper_right": 1,
    "upper_left": 2,
    "lower_left": 3,
    "lower_right": 4,
}


def _position(class_id: int):
    if class_id <= 0:
        return None
    for start, end, _, direction in _QUADRANT_DEFINITIONS:
        if start <= class_id <= end:
            offset = class_id - start
            return offset + 1 if direction == "incisor_to_molar" else 8 - offset
    return None


def _quadrant(class_id: int):
    for start, end, quadrant, _ in _QUADRANT_DEFINITIONS:
        if start <= class_id <= end:
            return quadrant
    return None


def _tooth_type(class_id: int):
    pos = _position(class_id)
    if pos is None:
        return None
    if pos in (1, 2):
        return "incisor"
    if pos == 3:
        return "canine"
    if pos in (4, 5):
        return "premolar"
    return "molar"


def tooth_meta(tooth_id: int) -> dict:
    """Anatomical labels for a tooth id (1..32). FDI is derived."""
    pos = _position(tooth_id)
    quad = _quadrant(tooth_id)
    fdi = None
    if pos is not None and quad in _FDI_QUADRANT_DIGIT:
        fdi = _FDI_QUADRANT_DIGIT[quad] * 10 + pos
    return {
        "tooth_id": tooth_id,
        "position": pos,
        "position_name": _POSITION_LABELS.get(pos),
        "quadrant": quad,
        "tooth_type": _tooth_type(tooth_id),
        "fdi": fdi,
    }


def build_table(prediction: Prediction) -> pd.DataFrame:
    rows = []
    for inst in prediction.instances:
        meta = tooth_meta(inst.tooth_id)
        rows.append(
            {
                "tooth_id": inst.tooth_id,
                "fdi": meta["fdi"],
                "class_name": meta["position_name"],
                "quadrant": meta["quadrant"],
                "tooth_type": meta["tooth_type"],
                "confidence": inst.confidence,
                "mask_area_pixels": inst.mask_area_pixels,
                "bbox_xyxy": inst.bbox_xyxy,
            }
        )
    columns = [
        "tooth_id", "fdi", "class_name", "quadrant", "tooth_type",
        "confidence", "mask_area_pixels", "bbox_xyxy",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values("tooth_id").reset_index(drop=True)
    return df


def build_report(prediction: Prediction, df: pd.DataFrame) -> dict:
    confs = [i.confidence for i in prediction.instances if i.confidence is not None]
    total_area = int(sum(i.mask_area_pixels for i in prediction.instances))
    image_area = int(prediction.height * prediction.width)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "AI Dental Panoramic Tooth Charting Assistant",
        "disclaimer": "Annotation/review aid only — not a diagnosis.",
        "model": {
            "name": prediction.model_name,
            "backend": prediction.backend,
            "params": prediction.params,
        },
        "image": {"height": prediction.height, "width": prediction.width},
        "summary": {
            "num_teeth_detected": prediction.num_teeth,
            "mean_confidence": round(sum(confs) / len(confs), 4) if confs else None,
            "total_tooth_area_pixels": total_area,
            "tooth_area_fraction": round(total_area / image_area, 4) if image_area else None,
            "fdi_present": sorted(int(x) for x in df["fdi"].dropna()) if not df.empty else [],
        },
        "teeth": df.to_dict(orient="records"),
    }


def report_json_bytes(report: dict) -> bytes:
    return json.dumps(report, indent=2).encode("utf-8")


def table_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
