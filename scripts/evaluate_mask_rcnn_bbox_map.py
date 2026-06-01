import argparse
import ctypes
import gc
import json
import os
import site
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from protocol_utils import write_json


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
FIXED_IMAGE_HEIGHT = 512
FIXED_IMAGE_WIDTH = 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Mask R-CNN checkpoints with ICPR-style class-wise bounding-box "
            "mAP@0.5 on validation folds."
        )
    )
    parser.add_argument("--fold", type=int, help="Fold index to evaluate. Omit with --all-folds.")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate every fold under splits-root/folds.")
    parser.add_argument("--subset", default="val", help="Fold subset to evaluate, usually val.")
    parser.add_argument("--mask-rcnn-root", type=Path, default=Path("src/mrcnn_tf2"))
    parser.add_argument("--splits-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--class-map", type=Path, default=Path("data/splits/class_map.txt"))
    parser.add_argument("--logs-dir", type=Path, default=Path("runs/mask_rcnn"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("best", "last"),
        default="best",
        help="Which checkpoint to evaluate inside each fold directory.",
    )
    parser.add_argument("--weights", type=Path, help="Explicit checkpoint path. Only valid with --fold.")
    return parser.parse_args()


def load_class_map(class_map_path):
    mapping = {}
    path = Path(class_map_path)
    if not path.exists():
        return mapping
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            idx = int(parts[0])
            name = parts[0]
        else:
            name = parts[0]
            idx = int(parts[-1])
        mapping[idx] = name
    return mapping


def list_image_mask_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    masks = {path.stem: path for path in masks_dir.glob("*.png")}
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    return [(path, masks[path.stem]) for path in sorted(image_paths) if path.stem in masks]


def discover_folds(splits_root):
    folds_root = Path(splits_root) / "folds"
    return sorted(
        int(path.name.split("_", 1)[1])
        for path in folds_root.glob("fold_*")
        if path.is_dir() and path.name.split("_", 1)[1].isdigit()
    )


def find_weights(logs_dir, fold, checkpoint_policy):
    fold_dir = Path(logs_dir) / f"fold_{fold}"
    if not fold_dir.exists():
        return None
    if checkpoint_policy == "best":
        candidates = list(fold_dir.rglob("mask_rcnn_teeth_best.h5"))
    else:
        candidates = [path for path in fold_dir.rglob("mask_rcnn_teeth_*.h5") if path.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def compute_box_overlaps(boxes1, boxes2):
    boxes1 = np.asarray(boxes1, dtype=np.float32).reshape(-1, 4)
    boxes2 = np.asarray(boxes2, dtype=np.float32).reshape(-1, 4)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0.0) * np.maximum(
        boxes1[:, 3] - boxes1[:, 1], 0.0
    )
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0.0) * np.maximum(
        boxes2[:, 3] - boxes2[:, 1], 0.0
    )
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    for i, box in enumerate(boxes1):
        y1 = np.maximum(box[0], boxes2[:, 0])
        x1 = np.maximum(box[1], boxes2[:, 1])
        y2 = np.minimum(box[2], boxes2[:, 2])
        x2 = np.minimum(box[3], boxes2[:, 3])
        intersection = np.maximum(y2 - y1, 0.0) * np.maximum(x2 - x1, 0.0)
        union = area1[i] + area2 - intersection
        overlaps[i] = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0,
        )
    return overlaps


def voc_ap(recalls, precisions):
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    return float(np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices]))


def compute_classwise_bbox_map(image_records, class_ids, iou_threshold):
    per_class = {}
    aps = []
    for class_id in class_ids:
        gt_by_image = {}
        num_gt = 0
        predictions = []
        for record in image_records:
            gt_keep = np.where(record["gt_class_ids"] == class_id)[0]
            class_gt_boxes = record["gt_boxes"][gt_keep]
            gt_by_image[record["image_key"]] = {
                "boxes": class_gt_boxes,
                "matched": np.zeros((class_gt_boxes.shape[0],), dtype=bool),
            }
            num_gt += int(class_gt_boxes.shape[0])

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
            per_class[int(class_id)] = {"ap": None, "num_gt": 0, "num_predictions": len(predictions)}
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

            overlaps = compute_box_overlaps(np.asarray([prediction["box"]]), gt_boxes)[0]
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
            ap = voc_ap(recalls, precisions)

        per_class[int(class_id)] = {
            "ap": float(ap),
            "num_gt": int(num_gt),
            "num_predictions": int(len(predictions)),
        }
        aps.append(float(ap))

    return float(np.mean(aps)) if aps else 0.0, per_class


def greedy_match(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    assign_gt_to_pred = -np.ones((gt_boxes.shape[0],), dtype=np.int32)
    assign_pred_to_gt = -np.ones((pred_boxes.shape[0],), dtype=np.int32)
    if pred_boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return assign_gt_to_pred, assign_pred_to_gt

    overlaps = compute_box_overlaps(pred_boxes, gt_boxes)
    for pred_idx in np.argsort(pred_scores)[::-1]:
        best_gt_idx = -1
        best_iou = -1.0
        for gt_idx in range(gt_boxes.shape[0]):
            if assign_gt_to_pred[gt_idx] >= 0:
                continue
            iou = float(overlaps[pred_idx, gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            assign_gt_to_pred[best_gt_idx] = pred_idx
            assign_pred_to_gt[pred_idx] = best_gt_idx
    return assign_gt_to_pred, assign_pred_to_gt


def make_table1_counts(image_records, num_classes, iou_threshold):
    tp = np.zeros((num_classes,), dtype=np.int64)
    fp = np.zeros((num_classes,), dtype=np.int64)
    fn = np.zeros((num_classes,), dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total = correct = misclassified = missed = false_positive = 0

    for record in image_records:
        gt_to_pred, pred_to_gt = greedy_match(
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


def ensure_cuda_runtime_paths():
    lib_dirs = []
    for root in site.getsitepackages():
        for path in glob(os.path.join(root, "nvidia", "*", "lib")):
            if path not in lib_dirs:
                lib_dirs.append(path)
    if not lib_dirs or os.environ.get("PBL4_NVRTC_REEXEC") == "1":
        return

    try:
        ctypes.CDLL("libnvrtc.so")
    except OSError:
        env = dict(os.environ)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + ([existing] if existing else []))
        env["PBL4_NVRTC_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def evaluate_fold(args, fold, class_map, modellib, utils, config_cls):
    split_root = Path(args.splits_root) / "folds" / f"fold_{fold}"
    pairs = list_image_mask_pairs(split_root / args.subset / "img", split_root / args.subset / "masks_semantic")
    if not pairs:
        raise SystemExit(f"No image/mask pairs found in {split_root / args.subset}")

    if args.weights is not None:
        if not Path(args.weights).exists():
            raise SystemExit(f"Checkpoint not found: {args.weights}")
        weights_path = Path(args.weights)
        checkpoint_selection = "explicit_path"
    else:
        weights_path = find_weights(args.logs_dir, fold, args.checkpoint_policy)
        checkpoint_selection = args.checkpoint_policy
    if weights_path is None:
        raise SystemExit(f"No Mask R-CNN checkpoint found for fold {fold}.")

    num_classes = max(class_map.keys()) + 1 if class_map else 1

    class FoldInferenceConfig(config_cls):
        NUM_CLASSES = num_classes
        DETECTION_MIN_CONFIDENCE = args.score_threshold

    config = FoldInferenceConfig()

    class TeethDataset(utils.Dataset):
        def load_teeth(self, pair_rows):
            for class_id, class_name in sorted(class_map.items()):
                if class_id == 0:
                    continue
                self.add_class("teeth", class_id, str(class_name))
            for image_path, mask_path in pair_rows:
                self.add_image(
                    "teeth",
                    image_id=image_path.stem,
                    path=str(image_path),
                    mask_path=str(mask_path),
                    width=FIXED_IMAGE_WIDTH,
                    height=FIXED_IMAGE_HEIGHT,
                )

        def load_image(self, image_id):
            info = self.image_info[image_id]
            image = Image.open(info["path"]).convert("RGB")
            image = image.resize((FIXED_IMAGE_WIDTH, FIXED_IMAGE_HEIGHT), resample=Image.BILINEAR)
            return np.asarray(image)

        def load_mask(self, image_id):
            info = self.image_info[image_id]
            mask = Image.open(info["mask_path"])
            mask = mask.resize((FIXED_IMAGE_WIDTH, FIXED_IMAGE_HEIGHT), resample=Image.NEAREST)
            mask = np.asarray(mask)
            if mask.ndim == 3:
                mask = mask[..., 0]
            class_ids = [int(class_id) for class_id in np.unique(mask) if int(class_id) > 0]
            if not class_ids:
                return np.zeros((mask.shape[0], mask.shape[1], 0), dtype=bool), np.array([], dtype=np.int32)
            masks = np.stack([(mask == class_id) for class_id in class_ids], axis=-1).astype(bool)
            return masks, np.asarray(class_ids, dtype=np.int32)

    dataset = TeethDataset()
    dataset.load_teeth(pairs)
    dataset.prepare()

    model_dir = Path(args.logs_dir) / f"fold_{fold}"
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=str(model_dir))
    model.load_weights(str(weights_path), by_name=True)

    image_records = []
    total_predictions = 0
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        gt_masks, gt_class_ids = dataset.load_mask(image_id)
        gt_boxes = utils.extract_bboxes(gt_masks) if gt_masks.size else np.zeros((0, 4), dtype=np.int32)

        result = model.detect([image], verbose=0)[0]
        pred_boxes = np.asarray(result["rois"], dtype=np.float32).reshape(-1, 4)
        pred_class_ids = np.asarray(result["class_ids"], dtype=np.int32).reshape(-1)
        pred_scores = np.asarray(result["scores"], dtype=np.float32).reshape(-1)

        keep = np.where(pred_scores >= args.score_threshold)[0]
        pred_boxes = pred_boxes[keep]
        pred_class_ids = pred_class_ids[keep]
        pred_scores = pred_scores[keep]
        total_predictions += int(pred_boxes.shape[0])

        image_records.append(
            {
                "image_key": dataset.image_info[image_id]["id"],
                "gt_boxes": np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4),
                "gt_class_ids": np.asarray(gt_class_ids, dtype=np.int32).reshape(-1),
                "pred_boxes": pred_boxes,
                "pred_class_ids": pred_class_ids,
                "pred_scores": pred_scores,
            }
        )

    class_ids = [class_id for class_id in sorted(class_map) if class_id > 0]
    map_score, per_class = compute_classwise_bbox_map(image_records, class_ids, args.iou_threshold)
    counts = make_table1_counts(image_records, num_classes, args.iou_threshold)

    summary = {
        "fold": int(fold),
        "metric_name": "bbox_mAP@0.5",
        "bbox_mAP@0.5": float(map_score),
        "metric_definition": (
            "Class-wise object-detection AP averaged over tooth classes, using predicted "
            "bounding boxes, ground-truth boxes extracted from semantic masks, and IoU=0.5."
        ),
        "subset": args.subset,
        "num_images": len(image_records),
        "num_gt_instances": int(sum(row["gt_class_ids"].shape[0] for row in image_records)),
        "num_predictions": int(total_predictions),
        "iou_threshold": float(args.iou_threshold),
        "score_threshold": float(args.score_threshold),
        "checkpoint_policy": checkpoint_selection,
        "weights_path": str(weights_path),
        "splits_root": str(args.splits_root),
        "class_ap": {
            str(class_id): {
                "class_name": str(class_map.get(class_id, class_id)),
                "ap": None if row["ap"] is None else float(row["ap"]),
                "num_gt": int(row["num_gt"]),
                "num_predictions": int(row["num_predictions"]),
            }
            for class_id, row in per_class.items()
        },
        "table1_style_counts": counts,
    }

    summary_path = Path(args.logs_dir) / f"cv_summary_fold_{fold}.json"
    write_json(summary_path, summary)
    return summary


def write_cv_summary(logs_dir, fold_summaries, expected_folds):
    merged = {}
    for summary_path in sorted(Path(logs_dir).glob("cv_summary_fold_*.json")):
        try:
            payload = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "bbox_mAP@0.5" not in payload:
            continue
        merged[int(payload["fold"])] = payload
    for row in fold_summaries:
        merged[int(row["fold"])] = row

    rows = [
        {
            "fold": int(row["fold"]),
            "bbox_mAP@0.5": float(row["bbox_mAP@0.5"]),
            "metric_name": row["metric_name"],
            "subset": row["subset"],
            "checkpoint_policy": row["checkpoint_policy"],
            "weights_path": row["weights_path"],
            "num_images": row["num_images"],
            "num_gt_instances": row["num_gt_instances"],
            "num_predictions": row["num_predictions"],
        }
        for row in sorted(merged.values(), key=lambda item: item["fold"])
    ]
    completed = [row["fold"] for row in rows]
    missing = [fold for fold in expected_folds if fold not in set(completed)]
    scores = [row["bbox_mAP@0.5"] for row in rows]
    summary = {
        "folds": rows,
        "aggregate": {
            "metric_name": "bbox_mAP@0.5",
            "metric_definition": (
                "Mean of class-wise bounding-box AP@0.5 over completed Mask R-CNN folds."
            ),
            "num_completed_folds": len(rows),
            "mean_bbox_mAP@0.5": float(np.mean(scores)) if scores else None,
            "std_bbox_mAP@0.5": float(np.std(scores)) if len(scores) > 1 else 0.0,
            "expected_folds": [int(fold) for fold in expected_folds],
            "missing_folds": [int(fold) for fold in missing],
            "is_complete": len(missing) == 0,
        },
    }
    summary_path = Path(logs_dir) / "cv_summary.json"
    write_json(summary_path, summary)
    return summary_path


def main():
    args = parse_args()
    if args.weights is not None and (args.all_folds or args.fold is None):
        raise SystemExit("--weights can only be used with one explicit --fold.")
    if not args.all_folds and args.fold is None:
        raise SystemExit("Pass --fold <index> or --all-folds.")

    project_root = Path(__file__).resolve().parents[1]
    mask_root = (project_root / args.mask_rcnn_root).resolve()
    if str(mask_root) not in sys.path:
        sys.path.insert(0, str(mask_root))

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    ensure_cuda_runtime_paths()

    from mrcnn.config import Config
    from mrcnn import model as modellib
    from mrcnn import utils
    import tensorflow as tf

    class TeethInferenceConfig(Config):
        NAME = "teeth"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1
        DETECTION_MIN_CONFIDENCE = 0.05
        BACKBONE = "resnet101"
        IMAGE_MIN_DIM = FIXED_IMAGE_HEIGHT
        IMAGE_MAX_DIM = FIXED_IMAGE_WIDTH
        IMAGE_RESIZE_MODE = "none"
        USE_MINI_MASK = False

    class_map = load_class_map(args.class_map)
    if not class_map:
        raise SystemExit(f"Class map is empty or missing: {args.class_map}")

    expected_folds = discover_folds(args.splits_root)
    folds = expected_folds if args.all_folds else [int(args.fold)]
    if args.all_folds and os.environ.get("PBL4_MRCNN_BBOX_MAP_CHILD") != "1":
        env = dict(os.environ)
        env["PBL4_MRCNN_BBOX_MAP_CHILD"] = "1"
        for fold in folds:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--fold",
                str(fold),
                "--subset",
                args.subset,
                "--mask-rcnn-root",
                str(args.mask_rcnn_root),
                "--splits-root",
                str(args.splits_root),
                "--class-map",
                str(args.class_map),
                "--logs-dir",
                str(args.logs_dir),
                "--iou-threshold",
                str(args.iou_threshold),
                "--score-threshold",
                str(args.score_threshold),
                "--checkpoint-policy",
                args.checkpoint_policy,
            ]
            subprocess.run(cmd, cwd=project_root, env=env, check=True)
        print(f"Wrote aggregate summary: {Path(args.logs_dir) / 'cv_summary.json'}")
        return

    summaries = []
    for fold in folds:
        print(f"Evaluating Mask R-CNN fold {fold} on {args.subset}...")
        summary = evaluate_fold(args, fold, class_map, modellib, utils, TeethInferenceConfig)
        summaries.append(summary)
        print(f"Fold {fold} bbox_mAP@0.5: {summary['bbox_mAP@0.5']:.4f}")
        tf.keras.backend.clear_session()
        gc.collect()

    summary_path = write_cv_summary(args.logs_dir, summaries, expected_folds)
    print(f"Wrote aggregate summary: {summary_path}")


if __name__ == "__main__":
    main()
