import argparse
import json
import os
import sys
import ctypes
import site
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from protocol_utils import write_json


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
FIXED_IMAGE_HEIGHT = 512
FIXED_IMAGE_WIDTH = 1024
DEFAULT_TOOTH_TYPES = {
    "incisor": [7, 8, 9, 10, 23, 24, 25, 26],
    "canine": [6, 11, 22, 27],
    "premolar": [4, 5, 12, 13, 20, 21, 28, 29],
    "molar": [1, 2, 3, 14, 15, 16, 17, 18, 19, 30, 31, 32],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Mask R-CNN like ICPR Table I / Fig.5 on a target split."
    )
    p.add_argument("--mode", choices=("eval",), default="eval")
    p.add_argument("--mask-rcnn-root", type=Path, default=Path("src/mrcnn_tf2"))
    p.add_argument("--splits-root", type=Path, default=Path("data/splits"))
    p.add_argument("--class-map", type=Path, default=Path("data/splits/class_map.txt"))
    p.add_argument("--logs-dir", type=Path, default=Path("runs/mask_rcnn"))
    p.add_argument("--fold", type=int, help="Fold model index. Default: best mAP fold from cv summary.")
    p.add_argument("--weights", type=Path, help="Optional explicit .h5 weights path.")
    p.add_argument(
        "--checkpoint-policy",
        choices=("best", "last"),
        default="best",
        help="Checkpoint selection policy when --weights is not provided.",
    )
    p.add_argument("--subset", default="test", help="Split subset to evaluate (default: test).")
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--score-threshold", type=float, default=0.05)
    p.add_argument("--out-dir", type=Path, default=Path("runs/mask_rcnn/icpr_eval"))
    p.add_argument(
        "--summary-tooth-type",
        type=Path,
        default=Path("runs/cv/summary_tooth_type_test.json"),
        help="Used to build Table II comparison with ICPR M-UNet.",
    )
    p.add_argument(
        "--munet-key",
        default="icpr_munet",
        help="Key in summary_tooth_type json used for ICPR M-UNet values.",
    )
    p.add_argument(
        "--allow-legacy-incompatible-comparison",
        action="store_true",
        help="Re-enable the historical detector-vs-segmentation comparison even though it mixes incompatible metrics.",
    )
    return p.parse_args()


def load_class_map(class_map_path: Path):
    mapping = {}
    if not class_map_path.exists():
        return mapping
    for line in class_map_path.read_text().splitlines():
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


def list_image_mask_pairs(images_dir: Path, masks_dir: Path):
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


def _pick_fold_from_cv_summary(logs_dir: Path):
    def _score(row):
        for key in ("bbox_mAP@0.5", "mAP@0.5"):
            value = row.get(key)
            if value is not None:
                return float(value)
        return -1.0

    summary_path = logs_dir / "cv_summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text())
            rows = payload.get("folds", [])
            if rows:
                best = max(rows, key=_score)
                return int(best["fold"])
        except Exception:
            pass
    best_fold = None
    best_map = -1.0
    for path in sorted(logs_dir.glob("cv_summary_fold_*.json")):
        try:
            payload = json.loads(path.read_text())
            fold = int(payload["fold"])
            m = _score(payload)
        except Exception:
            continue
        if m > best_map:
            best_map = m
            best_fold = fold
    return best_fold


def _find_weights(logs_dir: Path, fold: int):
    fold_dir = logs_dir / f"fold_{fold}"
    if not fold_dir.exists():
        return None
    best_files = list(fold_dir.rglob("mask_rcnn_teeth_best.h5"))
    if best_files:
        return sorted(best_files, key=lambda p: p.stat().st_mtime)[-1]
    return None


def _find_last_weights(logs_dir: Path, fold: int):
    fold_dir = logs_dir / f"fold_{fold}"
    if not fold_dir.exists():
        return None
    files = [p for p in fold_dir.rglob("mask_rcnn_teeth_*.h5") if p.is_file()]
    if not files:
        return None
    return sorted(files, key=lambda p: p.stat().st_mtime)[-1]


def _compute_overlaps(boxes1, boxes2):
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    area1 = np.maximum(boxes1[:, 2] - boxes1[:, 0], 0) * np.maximum(boxes1[:, 3] - boxes1[:, 1], 0)
    area2 = np.maximum(boxes2[:, 2] - boxes2[:, 0], 0) * np.maximum(boxes2[:, 3] - boxes2[:, 1], 0)

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    for i in range(boxes1.shape[0]):
        yi1 = np.maximum(boxes1[i, 0], boxes2[:, 0])
        xi1 = np.maximum(boxes1[i, 1], boxes2[:, 1])
        yi2 = np.minimum(boxes1[i, 2], boxes2[:, 2])
        xi2 = np.minimum(boxes1[i, 3], boxes2[:, 3])
        inter = np.maximum(yi2 - yi1, 0) * np.maximum(xi2 - xi1, 0)
        union = area1[i] + area2 - inter
        overlaps[i] = np.where(union > 0, inter / union, 0.0)
    return overlaps


def _greedy_match(pred_boxes, pred_scores, gt_boxes, iou_threshold):
    pred_n = pred_boxes.shape[0]
    gt_n = gt_boxes.shape[0]
    assign_gt_to_pred = -np.ones(gt_n, dtype=np.int32)
    assign_pred_to_gt = -np.ones(pred_n, dtype=np.int32)
    if pred_n == 0 or gt_n == 0:
        return assign_gt_to_pred, assign_pred_to_gt

    overlaps = _compute_overlaps(pred_boxes, gt_boxes)
    order = np.argsort(pred_scores)[::-1]
    for pi in order:
        best_gi = -1
        best_iou = -1.0
        for gi in range(gt_n):
            if assign_gt_to_pred[gi] != -1:
                continue
            iou = overlaps[pi, gi]
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            assign_gt_to_pred[best_gi] = pi
            assign_pred_to_gt[pi] = best_gi
    return assign_gt_to_pred, assign_pred_to_gt


def _render_table_png(title, rows, col_labels, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 2.0 + 0.35 * len(rows)))
    ax.axis("off")
    ax.set_title(title, fontsize=11, pad=8)
    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _render_confusion_submatrix_png(matrix, labels, title, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.imshow(matrix, cmap="Greens")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    vmax = matrix.max() if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            color = "white" if vmax > 0 and val > vmax * 0.55 else "black"
            ax.text(j, i, str(val), va="center", ha="center", color=color, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    mask_root = (project_root / args.mask_rcnn_root).resolve()
    if str(mask_root) not in sys.path:
        sys.path.insert(0, str(mask_root))

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    # Ensure pip-installed CUDA libs are visible (same pattern as train script).
    lib_dirs = []
    for root in site.getsitepackages():
        for path in glob(os.path.join(root, "nvidia", "*", "lib")):
            if path not in lib_dirs:
                lib_dirs.append(path)

    def _joined_lib_path():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        return ":".join(lib_dirs + ([existing] if existing else []))

    if lib_dirs and os.environ.get("PBL4_NVRTC_REEXEC") != "1":
        try:
            ctypes.CDLL("libnvrtc.so")
        except OSError:
            env = dict(os.environ)
            env["LD_LIBRARY_PATH"] = _joined_lib_path()
            env["PBL4_NVRTC_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable] + sys.argv, env)

    from mrcnn.config import Config
    from mrcnn import model as modellib
    from mrcnn import utils

    class_map = load_class_map(args.class_map)
    num_classes = max(class_map.keys()) + 1 if class_map else 1

    if args.fold is None:
        fold = _pick_fold_from_cv_summary(args.logs_dir)
        if fold is None:
            raise SystemExit("Unable to determine best fold from runs/mask_rcnn summaries.")
    else:
        fold = args.fold

    if args.weights is not None:
        weights_path = args.weights
        checkpoint_selection = "explicit_path"
    elif args.checkpoint_policy == "best":
        weights_path = _find_weights(args.logs_dir, fold)
        checkpoint_selection = "best"
    else:
        weights_path = _find_last_weights(args.logs_dir, fold)
        checkpoint_selection = "last"
    if weights_path is None or not Path(weights_path).exists():
        raise SystemExit(f"Missing weights for fold {fold}.")
    weights_path = Path(weights_path)

    class TeethConfig(Config):
        NAME = "teeth"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = num_classes
        DETECTION_MIN_CONFIDENCE = args.score_threshold
        IMAGE_MIN_DIM = FIXED_IMAGE_HEIGHT
        IMAGE_MAX_DIM = FIXED_IMAGE_WIDTH
        IMAGE_RESIZE_MODE = "none"
        USE_MINI_MASK = False

    class TeethDataset(utils.Dataset):
        def load_teeth(self, split_root, subset):
            for class_id, class_name in sorted(class_map.items()):
                if class_id == 0:
                    continue
                self.add_class("teeth", class_id, str(class_name))
            images_dir = split_root / subset / "img"
            masks_dir = split_root / subset / "masks_semantic"
            for image_path, mask_path in list_image_mask_pairs(images_dir, masks_dir):
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
            image = image.resize(
                (FIXED_IMAGE_WIDTH, FIXED_IMAGE_HEIGHT), resample=Image.BILINEAR
            )
            return np.array(image)

        def load_mask(self, image_id):
            info = self.image_info[image_id]
            mask = Image.open(info["mask_path"])
            mask = mask.resize(
                (FIXED_IMAGE_WIDTH, FIXED_IMAGE_HEIGHT), resample=Image.NEAREST
            )
            mask = np.array(mask)
            if mask.ndim == 3:
                mask = mask[..., 0]
            class_ids = [int(c) for c in np.unique(mask) if int(c) > 0]
            if not class_ids:
                return np.empty((mask.shape[0], mask.shape[1], 0), dtype=bool), np.array([], dtype=np.int32)
            masks = [(mask == c).astype(np.uint8) for c in class_ids]
            masks = np.stack(masks, axis=-1).astype(bool)
            return masks, np.array(class_ids, dtype=np.int32)

    dataset = TeethDataset()
    dataset.load_teeth(args.splits_root, args.subset)
    dataset.prepare()

    model_dir = args.logs_dir / f"fold_{fold}"
    model = modellib.MaskRCNN(mode="inference", config=TeethConfig(), model_dir=model_dir)
    model.load_weights(str(weights_path), by_name=True)

    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    total_teeth = 0
    correct = 0
    misclassified = 0
    missed = 0
    false_positive_detections = 0
    negative_images = 0
    negative_images_with_false_positives = 0
    true_negative_images = 0

    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        gt_masks, gt_class_ids = dataset.load_mask(image_id)
        gt_boxes = utils.extract_bboxes(gt_masks) if gt_masks.size else np.zeros((0, 4), dtype=np.int32)
        total_teeth += int(len(gt_class_ids))
        is_negative_image = len(gt_class_ids) == 0
        negative_images += int(is_negative_image)

        res = model.detect([image], verbose=0)[0]
        pred_boxes = res["rois"]
        pred_class_ids = res["class_ids"]
        pred_scores = res["scores"]

        keep = np.where(pred_scores >= args.score_threshold)[0]
        pred_boxes = pred_boxes[keep]
        pred_class_ids = pred_class_ids[keep]
        pred_scores = pred_scores[keep]

        gt_to_pred, pred_to_gt = _greedy_match(pred_boxes, pred_scores, gt_boxes, args.iou_threshold)

        for gi, gt_cls in enumerate(gt_class_ids):
            pi = gt_to_pred[gi]
            gt_cls = int(gt_cls)
            if pi < 0:
                missed += 1
                fn[gt_cls] += 1
                continue
            pred_cls = int(pred_class_ids[pi])
            confusion[gt_cls, pred_cls] += 1
            if pred_cls == gt_cls:
                correct += 1
                tp[gt_cls] += 1
            else:
                misclassified += 1
                fn[gt_cls] += 1
                fp[pred_cls] += 1

        for pi, gi in enumerate(pred_to_gt):
            if gi < 0:
                pred_cls = int(pred_class_ids[pi])
                fp[pred_cls] += 1
                false_positive_detections += 1

        if is_negative_image:
            if np.any(pred_to_gt < 0):
                negative_images_with_false_positives += 1
            else:
                true_negative_images += 1

    # Table I style
    table1 = {
        "fold_model": fold,
        "weights_path": str(weights_path),
        "checkpoint_selection": checkpoint_selection,
        "subset": args.subset,
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "total_number_of_teeth": int(total_teeth),
        "detected_and_correctly_classified": int(correct),
        "miss_classified_detections": int(misclassified),
        "missed_detections": int(missed),
        "false_positive_detections": int(false_positive_detections),
        "negative_images": int(negative_images),
        "negative_images_with_false_positives": int(negative_images_with_false_positives),
        "true_negative_images": int(true_negative_images),
    }

    # Detection-assignment F1 per class. This is not pixel Dice.
    class_assignment_f1 = {}
    for c in range(1, num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        class_assignment_f1[c] = float((2 * tp[c]) / denom) if denom > 0 else 0.0

    # Detector-only summary by tooth type from assignment F1.
    maskrcnn_tooth_type = {}
    for group, class_ids in DEFAULT_TOOTH_TYPES.items():
        vals = [class_assignment_f1[c] for c in class_ids if c in class_assignment_f1]
        maskrcnn_tooth_type[group] = {
            "class_ids": class_ids,
            "assignment_f1_mean": float(np.mean(vals)) if vals else 0.0,
        }

    comparison_reason = (
        "Disabled by default: detector assignment F1 is not comparable to segmentation pixel Dice, "
        "and the detector script selects a single fold while the segmentation summary is cross-fold aggregated."
    )
    table2 = {
        "status": "disabled",
        "reason": comparison_reason,
        "mask_rcnn_assignment_f1": maskrcnn_tooth_type,
        "icpr_munet_pixel_dice": {},
        "rows": [],
    }
    if args.allow_legacy_incompatible_comparison and args.summary_tooth_type.exists():
        payload = json.loads(args.summary_tooth_type.read_text())
        rows = payload.get(args.munet_key, [])
        by_group = {row["group"]: row for row in rows}
        table2["status"] = "legacy_incompatible"
        for group in ["incisor", "canine", "premolar", "molar"]:
            mod_dice = float(by_group[group]["dice_mean"]) if group in by_group else None
            mrcnn_f1 = float(maskrcnn_tooth_type[group]["assignment_f1_mean"])
            table2["icpr_munet_pixel_dice"][group] = mod_dice
            table2["rows"].append(
                {
                    "group": group,
                    "icpr_munet_pixel_dice": mod_dice,
                    "mask_rcnn_assignment_f1": mrcnn_f1,
                }
            )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "table1_detection_test.json", table1)
    write_json(out_dir / "detection_confusion_test.json", confusion.tolist())
    write_json(out_dir / "table2_compare_icprmunet_maskrcnn.json", table2)

    table1_rows = [
        ["Total Number of teeth", str(table1["total_number_of_teeth"])],
        ["Detected and correctly classified", str(table1["detected_and_correctly_classified"])],
        ["Miss-classified detections", str(table1["miss_classified_detections"])],
        ["Missed detections", str(table1["missed_detections"])],
        ["False-positive detections", str(table1["false_positive_detections"])],
        ["Negative images with false positives", str(table1["negative_images_with_false_positives"])],
    ]
    _render_table_png(
        title="Table I - Object detection results on test set (Mask R-CNN, IoU=0.5)",
        rows=table1_rows,
        col_labels=["Metric", "Count"],
        out_path=out_dir / "table1_detection_test.png",
    )

    molar_ids = DEFAULT_TOOTH_TYPES["molar"]
    sub = confusion[np.ix_(molar_ids, molar_ids)]
    labels = [f"T{class_map.get(c, c)}" for c in molar_ids]
    _render_confusion_submatrix_png(
        matrix=sub,
        labels=labels,
        title="Fig.5 - Molar sub-matrix (Mask R-CNN detection confusion, IoU=0.5)",
        out_path=out_dir / "fig5_molar_confusion_test.png",
    )

    if table2["rows"]:
        rows = []
        for r in table2["rows"]:
            rows.append(
                [
                    r["group"].capitalize(),
                    f"{r['icpr_munet_pixel_dice']*100:.2f}"
                    if r["icpr_munet_pixel_dice"] is not None
                    else "-",
                    f"{r['mask_rcnn_assignment_f1']*100:.2f}",
                ]
            )
        _render_table_png(
            title="Legacy incompatible comparison: pixel Dice vs detector assignment F1",
            rows=rows,
            col_labels=["Tooth type", "ICPR M-UNet pixel Dice", "Mask R-CNN assignment F1"],
            out_path=out_dir / "table2_compare_icprmunet_maskrcnn.png",
        )

    print("Wrote:", out_dir / "table1_detection_test.json")
    print("Wrote:", out_dir / "table1_detection_test.png")
    print("Wrote:", out_dir / "detection_confusion_test.json")
    print("Wrote:", out_dir / "fig5_molar_confusion_test.png")
    print("Wrote:", out_dir / "table2_compare_icprmunet_maskrcnn.json")
    if table2["rows"]:
        print("Wrote:", out_dir / "table2_compare_icprmunet_maskrcnn.png")
    else:
        print("Table II comparison disabled:", table2["reason"])


if __name__ == "__main__":
    main()
