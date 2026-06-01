"""Evaluate a trained YOLO-seg model on the fixed test split under the dense-
segmentation protocol.

Predicted instance masks are rasterized into a single 33-class label map, a
confusion matrix is accumulated against the (nearest-resized) GT semantic masks,
and the *same* reporting functions used by evaluate_final.py emit byte-compatible
artifacts next to the checkpoint::

    test_metrics.json           overall pixel-acc + macro/weighted IoU & Dice
    test_summary.json           macro/weighted/worst IoU & Dice  (== TransUNet)
    per_class_metrics_test.json
    per_position_metrics_test.json
    per_tooth_type_metrics_test.json
    per_quadrant_metrics_test.json
    test_evaluation_metadata.json

so the YOLO baselines drop straight into the comparison table alongside
TransUNet, mod_nestnet and icpr_munet.

Examples::

    python scripts/evaluate_yolo_seg.py --model yolo11 --cv-fold 0
    python scripts/evaluate_yolo_seg.py --checkpoint-path runs/cv/fold_0/yolo11_seg/weights/best.pt
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Pure-Python reporting helpers (no TF call path is exercised here).
from train import (
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    load_class_map,
    report_group_metrics,
    report_per_class_metrics,
)
from project_presets import (
    DEFAULT_YOLO_SEG_MODEL,
    YOLO_SEG_MODELS,
    get_yolo_seg_preset,
    resolve_yolo_seg_model,
)
from protocol_utils import summarize_mask_paths, write_json
from yolo_seg_utils import (
    list_image_mask_pairs,
    load_semantic_mask,
    rasterize_result,
)

SPLITS_DIR = Path(os.environ.get("PBL4_SPLITS_DIR", "data/splits"))
CLASS_MAP_PATH = Path(os.environ.get("PBL4_CLASS_MAP_PATH", "data/splits/class_map.txt"))
RUNS_DIR = Path("runs")
INCLUDE_BACKGROUND = False


def _infer_num_classes(class_map_path):
    max_id = 0
    for line in Path(class_map_path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            max_id = max(max_id, int(parts[-1]))
        except ValueError:
            continue
    return max_id + 1 if max_id else 1


def _summarize_overall(iou, dice, support, include_background):
    """Same overall summary shape as evaluate_final._summarize_overall."""
    start = 0 if include_background else 1
    idxs = list(range(start, len(iou)))
    if not idxs:
        return {}
    pixels = support[idxs]
    pixel_sum = int(pixels.sum())
    macro_iou = float(np.mean(iou[idxs]))
    macro_dice = float(np.mean(dice[idxs]))
    if pixel_sum > 0:
        weighted_iou = float(np.sum(iou[idxs] * pixels) / pixel_sum)
        weighted_dice = float(np.sum(dice[idxs] * pixels) / pixel_sum)
    else:
        weighted_iou = weighted_dice = 0.0
    worst_idx = min(idxs, key=lambda idx: iou[idx])
    return {
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "weighted_iou": weighted_iou,
        "weighted_dice": weighted_dice,
        "worst_class_id": int(worst_idx),
        "worst_iou": float(iou[worst_idx]),
        "worst_dice": float(dice[worst_idx]),
        "pixels": pixel_sum,
        "include_background": include_background,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_YOLO_SEG_MODEL,
                        help=f"Alias ({'/'.join(YOLO_SEG_MODELS)}) or weights spec.")
    parser.add_argument("--size", default=None, help="Backbone size used at train time.")
    parser.add_argument("--cv-fold", type=int, choices=range(4),
                        help="Evaluate runs/cv/fold_<n>/<run_name>/weights/best.pt.")
    parser.add_argument("--run-dir", type=Path, help="Explicit run dir holding weights/best.pt.")
    parser.add_argument("--checkpoint-path", type=Path, help="Explicit .pt checkpoint.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: preset).")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold (default: preset).")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _resolve_checkpoint(args, model_spec):
    if args.checkpoint_path is not None:
        ckpt = args.checkpoint_path
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found: {ckpt}")
        return ckpt, ckpt.parent.parent
    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.cv_fold is not None:
        run_dir = RUNS_DIR / "cv" / f"fold_{args.cv_fold}" / model_spec["run_name"]
    else:
        raise SystemExit("Provide one of --checkpoint-path, --run-dir or --cv-fold.")
    ckpt = run_dir / "weights" / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"No best.pt under {run_dir}/weights/. Train the fold first.")
    return ckpt, run_dir


def main():
    args = parse_args()
    preset = get_yolo_seg_preset()
    model_spec = resolve_yolo_seg_model(args.model, size=args.size)
    conf = args.conf if args.conf is not None else preset["conf_threshold"]
    iou_thr = args.iou if args.iou is not None else preset["iou_threshold"]
    imgsz = args.imgsz or preset["imgsz"]

    num_classes = _infer_num_classes(CLASS_MAP_PATH)
    height, width = DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH

    ckpt, out_dir = _resolve_checkpoint(args, model_spec)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_image_mask_pairs(SPLITS_DIR / "test" / "img",
                                  SPLITS_DIR / "test" / "masks_semantic")
    if not pairs:
        raise SystemExit("No test image/mask pairs found.")
    mask_stats = summarize_mask_paths([str(m) for _, m in pairs], num_classes, "test")

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    print(f"Evaluating {ckpt} on {len(pairs)} test images "
          f"(conf={conf}, iou={iou_thr}, imgsz={imgsz})")

    # Accumulate the confusion matrix one image at a time to bound memory.
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    flat = num_classes * num_classes
    for img_path, mask_path in pairs:
        result = model.predict(
            source=str(img_path), imgsz=imgsz, conf=conf, iou=iou_thr,
            device=args.device, verbose=False, retina_masks=True,
        )[0]
        pred = rasterize_result(result, height, width, num_classes, conf_threshold=conf)
        gt = load_semantic_mask(mask_path, height, width)
        idx = gt.reshape(-1) * num_classes + pred.reshape(-1)
        confusion += np.bincount(idx, minlength=flat).reshape(num_classes, num_classes)

    class_names = load_class_map(CLASS_MAP_PATH)
    class_names.setdefault(0, "background")

    iou, dice, support = report_per_class_metrics("test", confusion, class_names, out_dir)
    report_group_metrics("test", iou, dice, support, out_dir, num_classes)
    summary = _summarize_overall(iou, dice, support, INCLUDE_BACKGROUND)

    total = int(confusion.sum())
    pixel_acc = float(np.trace(confusion) / total) if total else 0.0
    metrics = {
        "pixel_accuracy": pixel_acc,
        "macro_iou": summary.get("macro_iou", 0.0),
        "macro_dice": summary.get("macro_dice", 0.0),
        "weighted_iou": summary.get("weighted_iou", 0.0),
        "weighted_dice": summary.get("weighted_dice", 0.0),
    }
    write_json(out_dir / "test_metrics.json", metrics)
    write_json(out_dir / "test_summary.json", summary)
    write_json(
        out_dir / "test_evaluation_metadata.json",
        {
            "run_dir": str(out_dir),
            "checkpoint": str(ckpt),
            "model": model_spec["alias"],
            "weights_run_name": model_spec["run_name"],
            "task": "instance_seg_rasterized_to_semantic",
            "conf_threshold": conf,
            "iou_threshold": iou_thr,
            "imgsz": imgsz,
            "eval_resolution": [height, width],
            "num_classes": int(num_classes),
            "test": {"samples": len(pairs), **mask_stats},
        },
    )
    print("Test metrics:", metrics)
    print(f"Wrote test summary to {out_dir / 'test_summary.json'}")


if __name__ == "__main__":
    main()
