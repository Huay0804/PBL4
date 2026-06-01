"""
Train or run YOLOX as a local alternative to the Mask R-CNN detector.

Train mode:
  - derives tooth boxes from existing semantic masks
  - materializes a fold-local COCO dataset view for YOLOX
  - launches the vendored YOLOX trainer

Detect mode:
  - loads a trained YOLOX checkpoint
  - runs inference on train/val/test subsets
  - exports class-wise BB priors compatible with the segmentation pipeline
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
import site
from glob import glob

import numpy as np
from PIL import Image

from project_presets import get_yolox_preset
from protocol_utils import (
    load_json,
    summarize_mask_paths,
    validate_bb_map_files,
    validate_disjoint_pair_sets,
    write_json,
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
SEED = 13
FIXED_IMAGE_HEIGHT = 512
FIXED_IMAGE_WIDTH = 1024

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
YOLOX_ROOT = PROJECT_ROOT / "src" / "yolox"
YOLOX_EXP_FILE = YOLOX_ROOT / "exps" / "pbl4" / "yolox_teeth_s.py"
SPLITS_DIR = Path(os.environ.get("PBL4_SPLITS_DIR", "data/splits"))
CLASS_MAP_PATH = Path(os.environ.get("PBL4_CLASS_MAP_PATH", "data/splits/class_map.txt"))
RUNS_DIR = Path(os.environ.get("PBL4_OUTPUT_DIR", "runs")) / "yolox"
YOLOX_BB_MAPS_ROOT = Path(os.environ.get("PBL4_YOLOX_BB_MAPS_DIR", "data/bb_maps/yolox"))
COCO_PRETRAINED_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
)


def load_class_map(class_map_path):
    path = Path(class_map_path)
    mapping = {}
    if not path.exists():
        return mapping
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


def list_image_mask_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    masks = {p.stem: p for p in masks_dir.glob("*.png")}
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    pairs = []
    for img in sorted(image_paths):
        mask = masks.get(img.stem)
        if mask is not None:
            pairs.append((img, mask))
    return pairs


def iter_image_paths(images_dir):
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(Path(images_dir).glob(f"*{ext}"))
    return sorted(image_paths)


def build_bb_map_from_boxes(rois, class_ids, scores, height, width, num_classes, min_score):
    bb_map = np.zeros((height, width, num_classes), dtype=np.uint8)
    best = {}
    for box, class_id, score in zip(rois, class_ids, scores):
        if score < min_score:
            continue
        if class_id not in best or score > best[class_id][0]:
            best[class_id] = (score, box)
    for class_id, (_score, box) in best.items():
        if class_id < 0 or class_id >= num_classes:
            continue
        y1, x1, y2, x2 = box
        bb_map[y1:y2, x1:x2, class_id] = 1
    return bb_map


def _xywh_to_yxyx(box):
    x1, y1, width, height = box
    return [float(y1), float(x1), float(y1 + height), float(x1 + width)]


def _compute_box_overlaps(pred_boxes, gt_boxes):
    if pred_boxes.size == 0 or gt_boxes.size == 0:
        return np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    pred_boxes = pred_boxes.astype(np.float32, copy=False)
    gt_boxes = gt_boxes.astype(np.float32, copy=False)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    overlaps = np.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=np.float32)

    for i, box in enumerate(pred_boxes):
        y1 = np.maximum(box[0], gt_boxes[:, 0])
        x1 = np.maximum(box[1], gt_boxes[:, 1])
        y2 = np.minimum(box[2], gt_boxes[:, 2])
        x2 = np.minimum(box[3], gt_boxes[:, 3])
        intersection = np.maximum(y2 - y1, 0.0) * np.maximum(x2 - x1, 0.0)
        union = pred_area[i] + gt_area - intersection
        overlaps[i] = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )

    return overlaps


def _compute_per_image_box_ap(gt_boxes, gt_class_ids, pred_boxes, pred_class_ids, pred_scores, iou_threshold=0.5):
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_class_ids = np.asarray(gt_class_ids, dtype=np.int32).reshape(-1)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    pred_class_ids = np.asarray(pred_class_ids, dtype=np.int32).reshape(-1)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)

    if gt_boxes.shape[0] == 0:
        return None

    if pred_boxes.shape[0] > 0:
        order = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[order]
        pred_class_ids = pred_class_ids[order]
        pred_scores = pred_scores[order]

    overlaps = _compute_box_overlaps(pred_boxes, gt_boxes)
    pred_match = -1 * np.ones((pred_boxes.shape[0],), dtype=np.int32)
    gt_match = -1 * np.ones((gt_boxes.shape[0],), dtype=np.int32)

    for i in range(pred_boxes.shape[0]):
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        for j in sorted_ixs:
            if gt_match[j] > -1:
                continue
            if overlaps[i, j] < iou_threshold:
                break
            if pred_class_ids[i] == gt_class_ids[j]:
                gt_match[j] = i
                pred_match[i] = j
                break

    precisions = np.cumsum(pred_match > -1) / (np.arange(pred_match.shape[0]) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / gt_match.shape[0]
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    recalls = np.concatenate([[0.0], recalls, [1.0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    return float(ap)


def _compute_voc_ap(recalls, precisions):
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    return float(np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices]))


def _compute_classwise_box_map(image_records, class_ids, iou_threshold=0.5):
    per_class = {}
    aps = []
    for class_id in class_ids:
        gt_by_image = {}
        predictions = []
        num_gt = 0

        for record in image_records:
            gt_keep = np.where(record["gt_class_ids"] == class_id)[0]
            gt_boxes = record["gt_boxes"][gt_keep]
            gt_by_image[record["image_key"]] = {
                "boxes": gt_boxes,
                "matched": np.zeros((gt_boxes.shape[0],), dtype=bool),
            }
            num_gt += int(gt_boxes.shape[0])

            pred_keep = np.where(record["pred_class_ids"] == class_id)[0]
            for pred_idx in pred_keep:
                predictions.append(
                    {
                        "image_key": record["image_key"],
                        "box": record["pred_boxes"][pred_idx],
                        "score": float(record["pred_scores"][pred_idx]),
                    }
                )

        if num_gt == 0:
            per_class[int(class_id)] = {
                "ap": None,
                "num_gt": 0,
                "num_predictions": len(predictions),
            }
            continue

        predictions.sort(key=lambda row: row["score"], reverse=True)
        tp = np.zeros((len(predictions),), dtype=np.float32)
        fp = np.zeros((len(predictions),), dtype=np.float32)

        for pred_idx, prediction in enumerate(predictions):
            gt_entry = gt_by_image[prediction["image_key"]]
            gt_boxes = gt_entry["boxes"]
            if gt_boxes.shape[0] == 0:
                fp[pred_idx] = 1.0
                continue

            overlaps = _compute_box_overlaps(np.asarray([prediction["box"]]), gt_boxes)[0]
            best_gt_idx = int(np.argmax(overlaps))
            best_iou = float(overlaps[best_gt_idx])
            if best_iou >= iou_threshold and not gt_entry["matched"][best_gt_idx]:
                tp[pred_idx] = 1.0
                gt_entry["matched"][best_gt_idx] = True
            else:
                fp[pred_idx] = 1.0

        if len(predictions) == 0:
            ap = 0.0
        else:
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / float(num_gt)
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float32).eps)
            ap = _compute_voc_ap(recalls, precisions)

        per_class[int(class_id)] = {
            "ap": float(ap),
            "num_gt": int(num_gt),
            "num_predictions": int(len(predictions)),
        }
        aps.append(float(ap))

    return float(np.mean(aps)) if aps else 0.0, per_class


def _greedy_match_boxes(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    gt_to_pred = -np.ones((gt_boxes.shape[0],), dtype=np.int32)
    pred_to_gt = -np.ones((pred_boxes.shape[0],), dtype=np.int32)
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return gt_to_pred, pred_to_gt

    overlaps = _compute_box_overlaps(pred_boxes, gt_boxes)
    for pred_idx in np.argsort(pred_scores)[::-1]:
        best_gt_idx = -1
        best_iou = -1.0
        for gt_idx in range(gt_boxes.shape[0]):
            if gt_to_pred[gt_idx] >= 0:
                continue
            iou = float(overlaps[pred_idx, gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_to_pred[best_gt_idx] = pred_idx
            pred_to_gt[pred_idx] = best_gt_idx
    return gt_to_pred, pred_to_gt


def _compute_table1_style_counts(image_records, num_classes, iou_threshold=0.5):
    tp = np.zeros((num_classes,), dtype=np.int64)
    fp = np.zeros((num_classes,), dtype=np.int64)
    fn = np.zeros((num_classes,), dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total = correct = misclassified = missed = false_positive = 0

    for record in image_records:
        gt_to_pred, pred_to_gt = _greedy_match_boxes(
            record["pred_boxes"],
            record["pred_scores"],
            record["gt_boxes"],
            iou_threshold,
        )
        total += int(record["gt_class_ids"].shape[0])
        for gt_idx, gt_class in enumerate(record["gt_class_ids"]):
            pred_idx = int(gt_to_pred[gt_idx])
            gt_class = int(gt_class)
            if pred_idx < 0:
                missed += 1
                fn[gt_class] += 1
                continue
            pred_class = int(record["pred_class_ids"][pred_idx])
            confusion[gt_class, pred_class] += 1
            if pred_class == gt_class:
                correct += 1
                tp[gt_class] += 1
            else:
                misclassified += 1
                fn[gt_class] += 1
                if 0 <= pred_class < num_classes:
                    fp[pred_class] += 1

        for pred_idx, gt_idx in enumerate(pred_to_gt):
            if gt_idx < 0:
                pred_class = int(record["pred_class_ids"][pred_idx])
                if 0 <= pred_class < num_classes:
                    fp[pred_class] += 1
                false_positive += 1

    return {
        "total_number_of_teeth": int(total),
        "detected_and_correctly_classified": int(correct),
        "miss_classified_detections": int(misclassified),
        "missed_detections": int(missed),
        "false_positive_detections": int(false_positive),
        "tp_by_class": {str(idx): int(value) for idx, value in enumerate(tp) if idx > 0},
        "fp_by_class": {str(idx): int(value) for idx, value in enumerate(fp) if idx > 0},
        "fn_by_class": {str(idx): int(value) for idx, value in enumerate(fn) if idx > 0},
        "confusion": confusion.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOX or export YOLOX BB priors for the segmentation pipeline."
    )
    parser.add_argument("--mode", choices=("train", "detect", "eval"), default="train")
    parser.add_argument("--fold", type=int, help="Training fold index used in train mode.")
    parser.add_argument("--source-fold", type=int, help="Detector fold used in detect mode.")
    parser.add_argument(
        "--target-fold",
        type=int,
        help="Target CV fold to export priors for in detect mode. Omit to export only the fixed test split.",
    )
    parser.add_argument(
        "--subset",
        choices=("train", "val", "test"),
        default="val",
        help="Subset used in eval mode.",
    )
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input-height", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--input-width", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num-workers", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--conf-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--nms-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--class-agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS during YOLOX inference. Default is class-aware NMS.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Training init weights: 'coco', 'none', or an explicit checkpoint path.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="YOLOX experiment name under the fold run directory.",
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=("best", "last"),
        default="best",
        help="Checkpoint selection policy used in detect mode when --checkpoint-path is not provided.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Explicit trained YOLOX checkpoint to use in detect mode.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of CUDA devices to use in YOLOX train mode.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Inference device used in detect mode.",
    )
    parser.add_argument(
        "--cache",
        choices=("ram", "disk"),
        nargs="?",
        const="ram",
        help="Enable YOLOX image caching in train mode.",
    )
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision where supported.")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="When exporting a target fold, also export priors for the fixed test split into the same root.",
    )
    args = parser.parse_args()

    preset = get_yolox_preset()
    if args.epochs is None:
        args.epochs = preset["epochs"]
    if args.batch_size is None:
        args.batch_size = preset["batch_size"]
    if args.learning_rate is None:
        args.learning_rate = preset["learning_rate"]
    if args.input_height is None:
        args.input_height = preset["input_height"]
    if args.input_width is None:
        args.input_width = preset["input_width"]
    if args.num_workers is None:
        args.num_workers = preset["num_workers"]
    if args.conf_threshold is None:
        args.conf_threshold = preset["conf_threshold"]
    if args.nms_threshold is None:
        args.nms_threshold = preset["nms_threshold"]
    if args.weights is None:
        args.weights = preset["weights"]
    if args.experiment_name is None:
        args.experiment_name = preset["experiment_name"]

    if args.mode == "train" and args.fold is None:
        raise SystemExit("--fold is required in train mode.")
    if args.mode in {"detect", "eval"} and args.source_fold is None:
        raise SystemExit("--source-fold is required in detect/eval mode.")
    return args


def _ensure_yolox_available():
    if not YOLOX_ROOT.exists():
        raise SystemExit(f"YOLOX root not found: {YOLOX_ROOT}")
    if not YOLOX_EXP_FILE.exists():
        raise SystemExit(f"YOLOX experiment file not found: {YOLOX_EXP_FILE}")


def _discover_cuda_lib_dirs():
    lib_dirs = []
    for root in site.getsitepackages():
        for path in glob(os.path.join(root, "nvidia", "*", "lib")):
            if path not in lib_dirs:
                lib_dirs.append(path)
    return lib_dirs


def _joined_lib_path(lib_dirs):
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return ":".join(lib_dirs + ([existing] if existing else []))


def _ensure_cuda_runtime_paths():
    lib_dirs = _discover_cuda_lib_dirs()
    if not lib_dirs:
        return lib_dirs
    if os.environ.get("PBL4_NVRTC_REEXEC") == "1":
        return lib_dirs

    probes = ("libnvrtc.so.13", "libnvrtc.so")
    try:
        for probe in probes:
            try:
                ctypes.CDLL(probe)
                return lib_dirs
            except OSError:
                continue
        raise OSError("NVRTC runtime libraries are not discoverable.")
    except OSError:
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = _joined_lib_path(lib_dirs)
        env["PBL4_NVRTC_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def _ensure_symlink(link_path, target_path):
    link_path = Path(link_path)
    target_path = Path(target_path).resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target_path:
            return
        link_path.unlink()
    elif link_path.exists():
        raise SystemExit(f"Expected generated symlink path, found existing directory/file: {link_path}")
    link_path.symlink_to(os.path.relpath(target_path, link_path.parent), target_is_directory=True)


def _mask_boxes(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    annotations = []
    for class_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        ys, xs = np.where(mask == class_id)
        if ys.size == 0:
            continue
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        annotations.append(
            {
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": int(ys.size),
            }
        )
    return annotations, mask.shape[0], mask.shape[1]


def _fixed_size_mask_boxes(mask_path):
    mask = Image.open(mask_path)
    mask = mask.resize((FIXED_IMAGE_WIDTH, FIXED_IMAGE_HEIGHT), resample=Image.NEAREST)
    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    annotations = []
    for class_id in sorted(int(v) for v in np.unique(mask) if int(v) != 0):
        ys, xs = np.where(mask == class_id)
        if ys.size == 0:
            continue
        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        annotations.append(
            {
                "category_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": int(ys.size),
            }
        )
    return annotations, FIXED_IMAGE_HEIGHT, FIXED_IMAGE_WIDTH


def _build_coco_annotations(subset_name, pairs, output_json, categories):
    images = []
    annotations = []
    annotation_id = 1
    category_ids = {category["id"] for category in categories}
    for image_id, (img_path, mask_path) in enumerate(pairs, start=1):
        subset_annotations, height, width = _mask_boxes(mask_path)
        images.append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": int(width),
                "height": int(height),
            }
        )
        for ann in subset_annotations:
            if ann["category_id"] not in category_ids:
                raise SystemExit(
                    f"Unexpected class id {ann['category_id']} in {mask_path}; update class_map.txt first."
                )
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "info": {"description": f"PBL4 teeth detection subset: {subset_name}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    write_json(output_json, payload)
    return {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "path": str(output_json),
    }


def _annotation_stats_from_existing(output_json):
    payload = load_json(output_json, default=None)
    if payload is None:
        return None
    return {
        "image_count": len(payload.get("images", [])),
        "annotation_count": len(payload.get("annotations", [])),
        "path": str(output_json),
        "reused": True,
    }


def _needs_annotation_rebuild(output_json, pairs):
    output_json = Path(output_json)
    if not output_json.exists():
        return True
    output_mtime = output_json.stat().st_mtime
    latest_input_mtime = max(
        max(img_path.stat().st_mtime, mask_path.stat().st_mtime)
        for img_path, mask_path in pairs
    )
    return latest_input_mtime > output_mtime


def prepare_yolox_coco_dataset(fold_index, mapping):
    fold_root = SPLITS_DIR / "folds" / f"fold_{fold_index}"
    if not fold_root.exists():
        raise SystemExit(f"Fold root not found: {fold_root}")

    class_ids = [cid for cid in sorted(mapping) if cid != 0]
    categories = [{"id": int(cid), "name": mapping[cid]} for cid in class_ids]
    coco_root = fold_root / "yolox_coco"
    annotations_root = coco_root / "annotations"

    subsets = [
        ("train", fold_root / "train" / "img", fold_root / "train" / "masks_semantic", "train2017", "instances_train2017.json"),
        ("val", fold_root / "val" / "img", fold_root / "val" / "masks_semantic", "val2017", "instances_val2017.json"),
    ]
    test_images_dir = SPLITS_DIR / "test" / "img"
    test_masks_dir = SPLITS_DIR / "test" / "masks_semantic"
    if test_images_dir.exists() and test_masks_dir.exists():
        subsets.append(("test", test_images_dir, test_masks_dir, "test2017", "instances_test2017.json"))

    annotation_stats = {}
    for subset_name, images_dir, masks_dir, image_link_name, ann_name in subsets:
        pairs = list_image_mask_pairs(images_dir, masks_dir)
        if not pairs:
            continue
        _ensure_symlink(coco_root / image_link_name, images_dir)
        annotation_path = annotations_root / ann_name
        if _needs_annotation_rebuild(annotation_path, pairs):
            annotation_stats[subset_name] = _build_coco_annotations(
                subset_name,
                pairs,
                annotation_path,
                categories,
            )
        else:
            annotation_stats[subset_name] = _annotation_stats_from_existing(annotation_path)

    return coco_root, annotation_stats


def _ensure_pretrained_weights(weights_arg):
    if weights_arg == "none":
        return None
    if weights_arg != "coco":
        resolved = Path(weights_arg)
        if not resolved.exists():
            raise SystemExit(f"Initial weights not found: {resolved}")
        return resolved

    destination = RUNS_DIR / "pretrained" / "yolox_s.pth"
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading YOLOX-S COCO weights to {destination}")
    urllib.request.urlretrieve(COCO_PRETRAINED_URL, destination)
    return destination


def _fold_logs_dir(fold_index):
    return RUNS_DIR / f"fold_{fold_index}"


def _make_run_name(base_name):
    return f"{base_name}{datetime.now().strftime('%Y%m%dT%H%M')}"


def _allocate_run_name(fold_logs_dir, base_name):
    fold_logs_dir = Path(fold_logs_dir)
    stem = _make_run_name(base_name)
    candidate = stem
    suffix = 1
    while (fold_logs_dir / candidate).exists():
        candidate = f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _iter_matching_run_dirs(fold_logs_dir, experiment_name=None):
    fold_logs_dir = Path(fold_logs_dir)
    if not fold_logs_dir.exists():
        return []
    run_dirs = [path for path in fold_logs_dir.iterdir() if path.is_dir()]
    if experiment_name:
        exact = [path for path in run_dirs if path.name == experiment_name]
        if exact:
            run_dirs = exact
        else:
            run_dirs = [path for path in run_dirs if path.name.startswith(experiment_name)]
    return sorted(run_dirs, key=lambda path: path.name)


def _resolve_matching_run_dirs(source_fold, experiment_name=None):
    fold_logs_dir = _fold_logs_dir(source_fold)
    run_dirs = _iter_matching_run_dirs(fold_logs_dir, experiment_name)
    if not run_dirs:
        raise SystemExit(
            f"No YOLOX run directories found in {fold_logs_dir}"
            + (f" matching '{experiment_name}'." if experiment_name else ".")
        )
    return fold_logs_dir, run_dirs


def _write_yolox_cv_summary(folds_root):
    per_fold = []
    for fold_path in sorted(RUNS_DIR.glob("cv_summary_fold_*.json")):
        try:
            payload = json.loads(fold_path.read_text())
            fold_idx = int(payload["fold"])
            map_score = float(payload["bbox_mAP@0.5"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        per_fold.append(
            {
                "fold": fold_idx,
                "bbox_mAP@0.5": map_score,
                "checkpoint_policy": payload.get("checkpoint_policy", "best"),
                "weights_path": payload.get("weights_path"),
                "run_dir": payload.get("run_dir"),
                "metric_name": payload.get("metric_name"),
                "subset": payload.get("subset", "val"),
                "conf_threshold": payload.get("conf_threshold"),
                "nms_threshold": payload.get("nms_threshold"),
                "class_agnostic_nms": payload.get("class_agnostic_nms"),
                "num_images": payload.get("num_images"),
                "num_gt_instances": payload.get("num_gt_instances"),
                "num_predictions": payload.get("num_predictions"),
            }
        )

    unique = {}
    for row in per_fold:
        unique[row["fold"]] = row
    rows = [unique[idx] for idx in sorted(unique.keys())]

    expected_folds = sorted(
        int(path.name.split("_")[1]) for path in folds_root.glob("fold_*") if path.is_dir()
    )
    completed_folds = [row["fold"] for row in rows]
    missing_folds = [idx for idx in expected_folds if idx not in set(completed_folds)]

    map_scores = [row["bbox_mAP@0.5"] for row in rows]
    aggregate = {
        "metric_name": "bbox_mAP@0.5",
        "metric_definition": "Mean of class-wise bounding-box AP@0.5 over completed YOLOX folds.",
        "num_completed_folds": len(rows),
        "mean_bbox_mAP@0.5": float(np.mean(map_scores)) if map_scores else None,
        "std_bbox_mAP@0.5": float(np.std(map_scores)) if len(map_scores) > 1 else 0.0,
        "expected_folds": expected_folds,
        "missing_folds": missing_folds,
        "is_complete": len(missing_folds) == 0,
    }

    summary = {"folds": rows, "aggregate": aggregate}
    summary_path = RUNS_DIR / "cv_summary.json"
    write_json(summary_path, summary)
    return summary_path


def _resolve_yolox_checkpoint(source_fold, experiment_name, checkpoint_policy, checkpoint_path=None):
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
        if not resolved.exists():
            raise SystemExit(f"Checkpoint not found: {resolved}")
        return resolved, {
            "requested_policy": "explicit",
            "selection": "explicit_path",
            "run_dir": str(resolved.parent),
        }

    fold_logs_dir, run_dirs = _resolve_matching_run_dirs(source_fold, experiment_name)
    if checkpoint_policy == "best":
        candidate_names = [("best_ckpt.pth", "best"), ("last_epoch_ckpt.pth", "best_fallback_last")]
    elif checkpoint_policy == "last":
        candidate_names = [("last_epoch_ckpt.pth", "last")]
    else:
        raise ValueError(f"Unknown checkpoint policy: {checkpoint_policy}")

    for filename, selection in candidate_names:
        candidates = [run_dir / filename for run_dir in run_dirs if (run_dir / filename).exists()]
        if candidates:
            resolved = max(candidates, key=lambda path: path.stat().st_mtime)
            return resolved, {
                "requested_policy": checkpoint_policy,
                "selection": selection,
                "run_dir": str(resolved.parent),
                "fold_dir": str(fold_logs_dir),
            }

    raise SystemExit(
        f"No YOLOX checkpoint found in {fold_logs_dir}"
        + (f" matching '{experiment_name}'" if experiment_name else "")
        + f" for policy '{checkpoint_policy}'."
    )


def _configure_yolox_imports():
    if str(YOLOX_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLOX_ROOT))
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


def _make_yolox_env(source_fold, coco_root, num_detector_classes, run_name, args):
    env = dict(os.environ)
    pythonpath = [str(YOLOX_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    lib_dirs = _discover_cuda_lib_dirs()
    if lib_dirs:
        env["LD_LIBRARY_PATH"] = _joined_lib_path(lib_dirs)
    env["PBL4_YOLOX_NUM_CLASSES"] = str(num_detector_classes)
    env["PBL4_YOLOX_DATA_DIR"] = str(coco_root)
    env["PBL4_YOLOX_OUTPUT_DIR"] = str(_fold_logs_dir(source_fold))
    env["PBL4_YOLOX_EXPERIMENT_NAME"] = run_name
    env["PBL4_YOLOX_INPUT_H"] = str(args.input_height)
    env["PBL4_YOLOX_INPUT_W"] = str(args.input_width)
    env["PBL4_YOLOX_NUM_WORKERS"] = str(args.num_workers)
    env["PBL4_YOLOX_EPOCHS"] = str(args.epochs)
    env["PBL4_YOLOX_EVAL_INTERVAL"] = "1"
    env["PBL4_YOLOX_BASIC_LR_PER_IMG"] = repr(args.learning_rate / float(args.batch_size))
    env["PBL4_YOLOX_TEST_CONF"] = repr(args.conf_threshold)
    env["PBL4_YOLOX_NMS"] = repr(args.nms_threshold)
    return env


class _LocalPredictor:
    def __init__(self, model, exp, device="cpu", fp16=False, class_agnostic_nms=False):
        from yolox.data.data_augment import ValTransform

        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.class_agnostic_nms = class_agnostic_nms
        self.preproc = ValTransform(legacy=False)

    def inference(self, image_path):
        import cv2
        import torch
        from yolox.utils import postprocess

        img = cv2.imread(str(image_path))
        if img is None:
            raise SystemExit(f"Failed to read image: {image_path}")
        height, width = img.shape[:2]
        ratio = min(self.test_size[0] / height, self.test_size[1] / width)
        tensor, _ = self.preproc(img, None, self.test_size)
        tensor = torch.from_numpy(tensor).unsqueeze(0).float()
        if self.device == "gpu":
            tensor = tensor.cuda()
            if self.fp16:
                tensor = tensor.half()

        with torch.no_grad():
            outputs = self.model(tensor)
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic=self.class_agnostic_nms,
            )
        return outputs, {"height": height, "width": width, "ratio": ratio}


def _load_predictor(source_fold, experiment_name, num_detector_classes, args):
    _configure_yolox_imports()
    import torch
    from yolox.exp import get_exp
    from yolox.utils import fuse_model

    env_backup = {
        "PBL4_YOLOX_NUM_CLASSES": os.environ.get("PBL4_YOLOX_NUM_CLASSES"),
        "PBL4_YOLOX_OUTPUT_DIR": os.environ.get("PBL4_YOLOX_OUTPUT_DIR"),
        "PBL4_YOLOX_EXPERIMENT_NAME": os.environ.get("PBL4_YOLOX_EXPERIMENT_NAME"),
        "PBL4_YOLOX_INPUT_H": os.environ.get("PBL4_YOLOX_INPUT_H"),
        "PBL4_YOLOX_INPUT_W": os.environ.get("PBL4_YOLOX_INPUT_W"),
        "PBL4_YOLOX_TEST_CONF": os.environ.get("PBL4_YOLOX_TEST_CONF"),
        "PBL4_YOLOX_NMS": os.environ.get("PBL4_YOLOX_NMS"),
    }
    os.environ["PBL4_YOLOX_NUM_CLASSES"] = str(num_detector_classes)
    os.environ["PBL4_YOLOX_OUTPUT_DIR"] = str(_fold_logs_dir(source_fold))
    os.environ["PBL4_YOLOX_EXPERIMENT_NAME"] = experiment_name
    os.environ["PBL4_YOLOX_INPUT_H"] = str(args.input_height)
    os.environ["PBL4_YOLOX_INPUT_W"] = str(args.input_width)
    os.environ["PBL4_YOLOX_TEST_CONF"] = repr(args.conf_threshold)
    os.environ["PBL4_YOLOX_NMS"] = repr(args.nms_threshold)
    try:
        exp = get_exp(str(YOLOX_EXP_FILE), None)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    model = exp.get_model()
    checkpoint_path, checkpoint_meta = _resolve_yolox_checkpoint(
        source_fold,
        experiment_name,
        args.checkpoint_policy,
        checkpoint_path=args.checkpoint_path,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available; rerun detect mode with --device cpu.")
        model.cuda()
        if args.fp16:
            model.half()
    model.eval()
    model = fuse_model(model)
    return (
        _LocalPredictor(
            model,
            exp,
            device=args.device,
            fp16=args.fp16,
            class_agnostic_nms=getattr(args, "class_agnostic_nms", False),
        ),
        checkpoint_path,
        checkpoint_meta,
    )


def _predict_boxes_for_image(predictor, img_path, detector_class_ids, min_score):
    outputs, info = predictor.inference(img_path)
    pred_boxes = []
    pred_class_ids = []
    pred_scores = []
    output = outputs[0]
    if output is None:
        return pred_boxes, pred_class_ids, pred_scores, info

    output = output.detach().cpu().numpy()
    boxes = output[:, 0:4] / info["ratio"]
    class_indices = output[:, 6].astype(np.int32)
    score_values = output[:, 4] * output[:, 5]
    for box, cls_index, score in zip(boxes, class_indices, score_values):
        if cls_index < 0 or cls_index >= len(detector_class_ids):
            continue
        score = float(score)
        if score < min_score:
            continue
        x1, y1, x2, y2 = box
        pred_boxes.append([float(y1), float(x1), float(y2), float(x2)])
        pred_class_ids.append(detector_class_ids[int(cls_index)])
        pred_scores.append(score)
    return pred_boxes, pred_class_ids, pred_scores, info


def _evaluate_predictor_on_pairs(predictor, detector_class_ids, pairs, min_score, subset_name):
    image_records = []
    images_without_gt = 0
    images_without_predictions = 0
    total_predictions = 0

    for img_path, mask_path in pairs:
        gt_annotations, _height, _width = _fixed_size_mask_boxes(mask_path)
        if not gt_annotations:
            images_without_gt += 1
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_class_ids = np.zeros((0,), dtype=np.int32)
        else:
            gt_boxes = np.asarray([_xywh_to_yxyx(ann["bbox"]) for ann in gt_annotations], dtype=np.float32)
            gt_class_ids = np.asarray([int(ann["category_id"]) for ann in gt_annotations], dtype=np.int32)

        pred_boxes, pred_class_ids, pred_scores, _info = _predict_boxes_for_image(
            predictor,
            img_path,
            detector_class_ids,
            min_score,
        )
        pred_boxes = [
            _scale_xyxy_box_to_fixed(
                [box[1], box[0], box[3], box[2]],
                _info["height"],
                _info["width"],
            )
            for box in pred_boxes
        ]
        if not pred_boxes:
            images_without_predictions += 1
        total_predictions += len(pred_boxes)
        image_records.append(
            {
                "image_key": img_path.stem,
                "gt_boxes": gt_boxes,
                "gt_class_ids": gt_class_ids,
                "pred_boxes": np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4),
                "pred_class_ids": np.asarray(pred_class_ids, dtype=np.int32).reshape(-1),
                "pred_scores": np.asarray(pred_scores, dtype=np.float32).reshape(-1),
            }
        )

    map_score, per_class = _compute_classwise_box_map(
        image_records,
        detector_class_ids,
        iou_threshold=0.5,
    )
    num_classes = max(detector_class_ids) + 1 if detector_class_ids else 1
    counts = _compute_table1_style_counts(image_records, num_classes, iou_threshold=0.5)

    return {
        "subset": subset_name,
        "metric_name": "bbox_mAP@0.5",
        "metric_note": (
            "Class-wise object-detection AP averaged over tooth classes, using predicted "
            "bounding boxes, ground-truth boxes extracted from semantic masks, and IoU=0.5. "
            "This matches the fixed Mask R-CNN detection summary protocol."
        ),
        "score": float(map_score),
        "bbox_mAP@0.5": float(map_score),
        "num_images": len(pairs),
        "num_eval_images": len(image_records),
        "num_gt_instances": int(sum(record["gt_class_ids"].shape[0] for record in image_records)),
            "num_predictions": int(total_predictions),
            "class_agnostic_nms": bool(getattr(predictor, "class_agnostic_nms", False)),
            "images_without_gt": images_without_gt,
            "images_without_predictions": images_without_predictions,
        "class_ap": {
            str(class_id): {
                "ap": None if row["ap"] is None else float(row["ap"]),
                "num_gt": int(row["num_gt"]),
                "num_predictions": int(row["num_predictions"]),
            }
            for class_id, row in per_class.items()
        },
        "table1_style_counts": counts,
    }


def _scale_xyxy_box_to_fixed(box, original_height, original_width):
    x1, y1, x2, y2 = box
    scale_y = FIXED_IMAGE_HEIGHT / float(original_height)
    scale_x = FIXED_IMAGE_WIDTH / float(original_width)
    y1 = int(np.floor(y1 * scale_y))
    x1 = int(np.floor(x1 * scale_x))
    y2 = int(np.ceil(y2 * scale_y))
    x2 = int(np.ceil(x2 * scale_x))
    y1 = max(0, min(FIXED_IMAGE_HEIGHT, y1))
    x1 = max(0, min(FIXED_IMAGE_WIDTH, x1))
    y2 = max(0, min(FIXED_IMAGE_HEIGHT, y2))
    x2 = max(0, min(FIXED_IMAGE_WIDTH, x2))
    return [y1, x1, y2, x2]


def _export_subset_bb_maps(predictor, detector_class_ids, subset_name, images_dir, output_dir, num_classes, min_score):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = iter_image_paths(images_dir)
    if not image_paths:
        raise SystemExit(f"No images found for subset '{subset_name}' in {images_dir}")

    nonzero = 0
    for img_path in image_paths:
        pred_boxes, class_ids, scores, info = _predict_boxes_for_image(
            predictor,
            img_path,
            detector_class_ids,
            min_score,
        )
        rois = [
            _scale_xyxy_box_to_fixed([box[1], box[0], box[3], box[2]], info["height"], info["width"])
            for box in pred_boxes
        ]

        bb_map = build_bb_map_from_boxes(
            rois,
            class_ids,
            scores,
            FIXED_IMAGE_HEIGHT,
            FIXED_IMAGE_WIDTH,
            num_classes,
            min_score,
        )
        nonzero += int(np.any(bb_map))
        np.savez_compressed(output_dir / f"{img_path.stem}.npz", bb=bb_map)

    bb_paths = sorted(output_dir.glob("*.npz"))
    stats = validate_bb_map_files(
        bb_paths,
        FIXED_IMAGE_HEIGHT,
        FIXED_IMAGE_WIDTH,
        num_classes,
        subset_name=subset_name,
    )
    stats["nonzero_files"] = nonzero
    return stats


def run_train(args):
    _ensure_yolox_available()
    mapping = load_class_map(CLASS_MAP_PATH)
    if not mapping:
        raise SystemExit(f"Class map not found or empty: {CLASS_MAP_PATH}")
    detector_class_ids = [cid for cid in sorted(mapping) if cid != 0]
    num_classes = max(mapping) + 1
    fold_root = SPLITS_DIR / "folds" / f"fold_{args.fold}"
    train_pairs = list_image_mask_pairs(fold_root / "train" / "img", fold_root / "train" / "masks_semantic")
    val_pairs = list_image_mask_pairs(fold_root / "val" / "img", fold_root / "val" / "masks_semantic")
    test_pairs = list_image_mask_pairs(SPLITS_DIR / "test" / "img", SPLITS_DIR / "test" / "masks_semantic")
    if not train_pairs or not val_pairs:
        raise SystemExit(f"Training fold {args.fold} is missing train/val image-mask pairs.")
    validate_disjoint_pair_sets(
        {
            "train": train_pairs,
            "val": val_pairs,
            **({"test": test_pairs} if test_pairs else {}),
        }
    )

    train_mask_stats = summarize_mask_paths([pair[1] for pair in train_pairs], num_classes, "train")
    val_mask_stats = summarize_mask_paths([pair[1] for pair in val_pairs], num_classes, "val")
    test_mask_stats = summarize_mask_paths([pair[1] for pair in test_pairs], num_classes, "test") if test_pairs else None
    coco_root, annotation_stats = prepare_yolox_coco_dataset(args.fold, mapping)
    weights_path = _ensure_pretrained_weights(args.weights)
    fold_logs_dir = _fold_logs_dir(args.fold)
    fold_logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = _allocate_run_name(fold_logs_dir, args.experiment_name)
    env = _make_yolox_env(args.fold, coco_root, len(detector_class_ids), run_name, args)

    command = [
        sys.executable,
        str(YOLOX_ROOT / "tools" / "train.py"),
        "-f",
        str(YOLOX_EXP_FILE),
        "-expn",
        run_name,
        "-d",
        str(args.devices),
        "-b",
        str(args.batch_size),
        "--logger",
        "tensorboard",
    ]
    if weights_path is not None:
        command.extend(["-c", str(weights_path)])
    if args.cache:
        command.extend(["--cache", args.cache])
    if args.fp16:
        command.append("--fp16")

    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)

    run_root = fold_logs_dir / run_name
    predictor, checkpoint_path, checkpoint_meta = _load_predictor(
        args.fold,
        run_name,
        len(detector_class_ids),
        args,
    )
    val_eval = _evaluate_predictor_on_pairs(
        predictor,
        detector_class_ids,
        val_pairs,
        args.conf_threshold,
        "val",
    )
    summary = {
        "mode": "train",
        "fold": args.fold,
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "run_root": str(run_root),
        "command": [str(part) for part in command],
        "weights": str(weights_path) if weights_path is not None else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_meta": checkpoint_meta,
        "input_size": [args.input_height, args.input_width],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "num_classes": len(detector_class_ids),
        "mask_stats": {
            "train": train_mask_stats,
            "val": val_mask_stats,
            **({"test": test_mask_stats} if test_mask_stats is not None else {}),
        },
        "annotations": annotation_stats,
        "evaluation": val_eval,
        "seed": SEED,
    }
    write_json(run_root / "train_summary.json", summary)

    fold_summary = {
        "fold": args.fold,
        "bbox_mAP@0.5": val_eval["score"],
        "metric_name": val_eval["metric_name"],
        "metric_note": val_eval["metric_note"],
        "subset": val_eval["subset"],
        "conf_threshold": args.conf_threshold,
        "nms_threshold": args.nms_threshold,
        "class_agnostic_nms": val_eval["class_agnostic_nms"],
        "num_images": val_eval["num_images"],
        "num_gt_instances": val_eval["num_gt_instances"],
        "num_predictions": val_eval["num_predictions"],
        "class_ap": val_eval["class_ap"],
        "table1_style_counts": val_eval["table1_style_counts"],
        "checkpoint_policy": "best",
        "weights_path": str(checkpoint_path),
        "run_dir": str(run_root),
        "experiment_name": args.experiment_name,
        "run_name": run_name,
        "seed": SEED,
    }
    summary_path = RUNS_DIR / f"cv_summary_fold_{args.fold}.json"
    write_json(summary_path, fold_summary)
    print(f"Fold {args.fold} bbox_mAP@0.5: {val_eval['score']:.4f}")
    print(f"Wrote fold summary to {summary_path}")
    cv_summary_path = _write_yolox_cv_summary(SPLITS_DIR / 'folds')
    print(f"Updated CV summary at {cv_summary_path}")


def run_eval(args):
    _ensure_yolox_available()
    mapping = load_class_map(CLASS_MAP_PATH)
    if not mapping:
        raise SystemExit(f"Class map not found or empty: {CLASS_MAP_PATH}")
    detector_class_ids = [cid for cid in sorted(mapping) if cid != 0]

    if args.subset == "test":
        pairs = list_image_mask_pairs(SPLITS_DIR / "test" / "img", SPLITS_DIR / "test" / "masks_semantic")
        target_fold = None
    else:
        fold = args.source_fold if args.target_fold is None else args.target_fold
        subset_root = SPLITS_DIR / "folds" / f"fold_{fold}" / args.subset
        pairs = list_image_mask_pairs(subset_root / "img", subset_root / "masks_semantic")
        target_fold = fold
    if not pairs:
        raise SystemExit(f"No image-mask pairs found for eval subset '{args.subset}'.")

    predictor, checkpoint_path, checkpoint_meta = _load_predictor(
        args.source_fold,
        args.experiment_name,
        len(detector_class_ids),
        args,
    )
    eval_summary = _evaluate_predictor_on_pairs(
        predictor,
        detector_class_ids,
        pairs,
        args.conf_threshold,
        args.subset,
    )
    eval_summary.update(
        {
            "mode": "eval",
            "source_fold": args.source_fold,
            "target_fold": target_fold,
            "experiment_name": args.experiment_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_meta": checkpoint_meta,
            "input_size": [args.input_height, args.input_width],
            "conf_threshold": args.conf_threshold,
            "nms_threshold": args.nms_threshold,
            "class_agnostic_nms": args.class_agnostic_nms,
        }
    )

    out_name = f"eval_{args.subset}.json" if target_fold is None else f"eval_target_fold_{target_fold}_{args.subset}.json"
    if checkpoint_meta.get("run_dir"):
        out_path = Path(checkpoint_meta["run_dir"]) / out_name
    else:
        out_path = _fold_logs_dir(args.source_fold) / out_name
    write_json(out_path, eval_summary)
    print(f"{eval_summary['metric_name']} ({args.subset}): {eval_summary['score']:.4f}")
    print(f"Wrote evaluation summary to {out_path}")


def run_detect(args):
    _ensure_yolox_available()
    mapping = load_class_map(CLASS_MAP_PATH)
    if not mapping:
        raise SystemExit(f"Class map not found or empty: {CLASS_MAP_PATH}")
    detector_class_ids = [cid for cid in sorted(mapping) if cid != 0]
    num_classes = max(mapping) + 1

    predictor, checkpoint_path, checkpoint_meta = _load_predictor(
        args.source_fold,
        args.experiment_name,
        len(detector_class_ids),
        args,
    )

    export_root = YOLOX_BB_MAPS_ROOT
    if args.target_fold is None:
        subsets = [("test", SPLITS_DIR / "test" / "img", export_root / "test" / "bb_maps")]
    else:
        target_splits_root = SPLITS_DIR / "folds" / f"fold_{args.target_fold}"
        if not target_splits_root.exists():
            raise SystemExit(f"Target fold root not found: {target_splits_root}")
        target_export_root = export_root / "folds" / f"fold_{args.target_fold}"
        subsets = [
            ("train", target_splits_root / "train" / "img", target_export_root / "train" / "bb_maps"),
            ("val", target_splits_root / "val" / "img", target_export_root / "val" / "bb_maps"),
        ]
        if args.include_test:
            subsets.append(("test", SPLITS_DIR / "test" / "img", export_root / "test" / "bb_maps"))

    for subset_name, images_dir, output_dir in subsets:
        subset_stats = _export_subset_bb_maps(
            predictor,
            detector_class_ids,
            subset_name,
            images_dir,
            output_dir,
            num_classes,
            args.conf_threshold,
        )
        write_json(
            Path(output_dir) / "_export_metadata.json",
            {
                "subset": subset_name,
                "splits_root": str(
                    target_splits_root if subset_name != "test" and args.target_fold is not None else SPLITS_DIR
                ),
                "export_root": str(export_root),
                "source_model_dir": str(_fold_logs_dir(args.source_fold)),
                "source_run_dir": checkpoint_meta.get("run_dir"),
                "source_fold": args.source_fold,
                "target_fold": args.target_fold,
                "checkpoint_policy": args.checkpoint_policy,
                "checkpoint_selection": checkpoint_meta.get("selection"),
                "weights_path": str(checkpoint_path),
                "class_map_path": str(CLASS_MAP_PATH),
                "num_classes": int(num_classes),
                "bb_map_shape": [FIXED_IMAGE_HEIGHT, FIXED_IMAGE_WIDTH, num_classes],
                "images": subset_stats.get("count", subset_stats.get("num_files")),
                "nonzero_bb_maps": subset_stats.get("nonzero_files"),
                "detection_min_confidence": args.conf_threshold,
                "nms_threshold": args.nms_threshold,
                "class_agnostic_nms": args.class_agnostic_nms,
                "seed": SEED,
            },
        )
    print(f"Saved BB maps under {export_root}.")


def main():
    args = parse_args()
    _ensure_cuda_runtime_paths()
    if args.mode == "train":
        _configure_yolox_imports()
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("YOLOX train mode requires CUDA; no GPU was detected.")
        if args.devices < 1:
            raise SystemExit("--devices must be at least 1 in train mode.")
        if args.devices > torch.cuda.device_count():
            raise SystemExit(
                f"Requested {args.devices} CUDA devices, but only {torch.cuda.device_count()} are available."
            )
        run_train(args)
    elif args.mode == "detect":
        _configure_yolox_imports()
        if args.device == "gpu":
            import torch

            if not torch.cuda.is_available():
                raise SystemExit("CUDA is not available; rerun detect mode with --device cpu.")
        run_detect(args)
    else:
        _configure_yolox_imports()
        if args.device == "gpu":
            import torch

            if not torch.cuda.is_available():
                raise SystemExit("CUDA is not available; rerun eval mode with --device cpu.")
        run_eval(args)


if __name__ == "__main__":
    main()
