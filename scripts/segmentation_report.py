"""Dependency-free segmentation metric reporting.

These helpers are mirrored verbatim from train.py so the YOLO-seg evaluation can
produce byte-compatible per-class / per-position / per-tooth-type / per-quadrant
metrics *without* importing train.py (which pulls in TensorFlow, Keras and the
segmentation_models package). Keeping this module pure — only json, numpy and
pathlib — lets evaluate_yolo_seg.py run on a torch-only Ultralytics environment
(e.g. the Kaggle YOLO notebook) where TensorFlow may be unavailable or ABI-broken
after a pip install.

If the scoring logic in train.py changes, update it here too.
"""

import json
from pathlib import Path

import numpy as np


DEFAULT_INPUT_HEIGHT = 512
DEFAULT_INPUT_WIDTH = 1024


def load_class_map(class_map_path):
    path = Path(class_map_path)
    if not path.exists():
        return {}
    mapping = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            name = parts[0]
            idx = int(parts[0])
        else:
            name = parts[0]
            idx = int(parts[-1])
        mapping[idx] = name
    return mapping


def _safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def per_class_metrics_from_confusion(confusion):
    true_positives = np.diag(confusion)
    false_positives = confusion.sum(axis=0) - true_positives
    false_negatives = confusion.sum(axis=1) - true_positives

    iou = _safe_divide(true_positives, true_positives + false_positives + false_negatives)
    dice = _safe_divide(
        2.0 * true_positives,
        2.0 * true_positives + false_positives + false_negatives,
    )
    support = confusion.sum(axis=1)
    return iou, dice, support


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


def _class_id_to_position(class_id):
    if class_id <= 0:
        return None
    for start, end, _, direction in _QUADRANT_DEFINITIONS:
        if start <= class_id <= end:
            offset = class_id - start
            if direction == "incisor_to_molar":
                return offset + 1
            return 8 - offset
    return None


def _class_id_to_quadrant(class_id):
    if class_id <= 0:
        return None
    for start, end, quadrant, _ in _QUADRANT_DEFINITIONS:
        if start <= class_id <= end:
            return quadrant
    return None


def _class_id_to_tooth_type(class_id):
    position = _class_id_to_position(class_id)
    if position is None:
        return None
    if position in (1, 2):
        return "incisor"
    if position == 3:
        return "canine"
    if position in (4, 5):
        return "premolar"
    return "molar"


def _summarize_groups(groups, iou, dice, support):
    rows = []
    for name, class_ids in groups.items():
        values_iou = [iou[class_id] for class_id in class_ids]
        values_dice = [dice[class_id] for class_id in class_ids]
        pixels = int(sum(support[class_id] for class_id in class_ids))
        rows.append(
            {
                "group": name,
                "class_ids": class_ids,
                "iou_mean": float(np.mean(values_iou)),
                "iou_std": float(np.std(values_iou)),
                "dice_mean": float(np.mean(values_dice)),
                "dice_std": float(np.std(values_dice)),
                "pixels": pixels,
            }
        )
    return rows


def report_per_class_metrics(name, confusion, class_names, out_dir):
    iou, dice, support = per_class_metrics_from_confusion(confusion)
    rows = []
    for class_id, (iou_val, dice_val, count) in enumerate(zip(iou, dice, support)):
        class_name = class_names.get(class_id, str(class_id))
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "iou": float(iou_val),
                "dice": float(dice_val),
                "pixels": int(count),
            }
        )
    rows.sort(key=lambda r: r["iou"])
    out_path = Path(out_dir) / f"per_class_metrics_{name}.json"
    out_path.write_text(json.dumps(rows, indent=2))

    worst = [r for r in rows if r["class_id"] != 0][:5]
    print(f"Per-class {name} metrics (worst IoU):")
    for row in worst:
        print(
            f"  class {row['class_id']} ({row['class_name']}): "
            f"IoU={row['iou']:.4f} Dice={row['dice']:.4f} pixels={row['pixels']}"
        )
    return iou, dice, support


def report_group_metrics(name, iou, dice, support, out_dir, num_classes):
    if num_classes - 1 != 32:
        print("Skipping per-position metrics (expected 32 tooth classes).")
        return

    position_groups = {}
    for class_id in range(1, num_classes):
        position = _class_id_to_position(class_id)
        if position is None:
            continue
        position_groups.setdefault(position, []).append(class_id)

    tooth_groups = {"incisor": [], "canine": [], "premolar": [], "molar": []}
    for class_id in range(1, num_classes):
        tooth_type = _class_id_to_tooth_type(class_id)
        if tooth_type is not None:
            tooth_groups[tooth_type].append(class_id)

    labeled_positions = {}
    for pos, class_ids in position_groups.items():
        label = _POSITION_LABELS.get(pos, f"pos_{pos}")
        labeled_positions[f"{pos}_{label}"] = class_ids

    position_rows = _summarize_groups(labeled_positions, iou, dice, support)
    position_rows.sort(key=lambda row: row["group"])
    (Path(out_dir) / f"per_position_metrics_{name}.json").write_text(
        json.dumps(position_rows, indent=2)
    )

    tooth_rows = _summarize_groups(tooth_groups, iou, dice, support)
    (Path(out_dir) / f"per_tooth_type_metrics_{name}.json").write_text(
        json.dumps(tooth_rows, indent=2)
    )

    quadrant_groups = {}
    for class_id in range(1, num_classes):
        quadrant = _class_id_to_quadrant(class_id)
        if quadrant is None:
            continue
        quadrant_groups.setdefault(quadrant, []).append(class_id)
    quadrant_rows = _summarize_groups(quadrant_groups, iou, dice, support)
    quadrant_rows.sort(key=lambda row: row["group"])
    (Path(out_dir) / f"per_quadrant_metrics_{name}.json").write_text(
        json.dumps(quadrant_rows, indent=2)
    )

    print(f"Per-position {name} metrics saved to per_position_metrics_{name}.json")
    print(f"Per-tooth-type {name} metrics saved to per_tooth_type_metrics_{name}.json")
    print(f"Per-quadrant {name} metrics saved to per_quadrant_metrics_{name}.json")
