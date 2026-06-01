"""
Train a TF2-compatible Mask R-CNN implementation for teeth detection.
Edit MODE below to switch between training and BB-map export.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from project_presets import get_mask_rcnn_preset
from protocol_utils import summarize_mask_paths, validate_disjoint_pair_sets, write_json

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

SPLITS_DIR = Path(os.environ.get("PBL4_SPLITS_DIR", "data/splits"))
CLASS_MAP_PATH = Path(os.environ.get("PBL4_CLASS_MAP_PATH", "data/splits/class_map.txt"))
LOGS_DIR = Path("runs/mask_rcnn")
MASK_RCNN_BB_MAPS_ROOT = Path(os.environ.get("PBL4_MASK_RCNN_BB_MAPS_DIR", "data/bb_maps/mask_rcnn"))
WEIGHTS = "coco"
DETECT_CHECKPOINT_POLICY = "best"
# Default to fold-based training to keep outputs under runs/mask_rcnn/fold_<k>.
MODE = "train"  # "train" or "detect"
TRAIN_SUBSET = "train"
VAL_SUBSET = "val"
DETECT_SUBSET = "test"
DETECTION_MIN_CONFIDENCE = 0.05
MAP_IOU_THRESHOLD = 0.5
FIT_MAX_QUEUE_SIZE = 10
# ICPR input size (512x1024). Keep fixed to avoid shape mismatches.
DEFAULT_IMAGE_MIN_DIM = 512
DEFAULT_IMAGE_MAX_DIM = 1024
FIXED_IMAGE_HEIGHT = DEFAULT_IMAGE_MIN_DIM
FIXED_IMAGE_WIDTH = DEFAULT_IMAGE_MAX_DIM
SEED = 13


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


def build_bb_map_from_boxes(rois, class_ids, scores, height, width, num_classes, min_score):
    bb_map = np.zeros((height, width, num_classes), dtype=np.uint8)
    best = {}
    for box, class_id, score in zip(rois, class_ids, scores):
        if score < min_score:
            continue
        if class_id not in best or score > best[class_id][0]:
            best[class_id] = (score, box)
    for class_id, (_score, box) in best.items():
        y1, x1, y2, x2 = box
        bb_map[y1:y2, x1:x2, class_id] = 1
    return bb_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN or export fold-specific BB maps using project presets."
    )
    parser.add_argument("--mode", choices=("train", "detect"), default=MODE)
    parser.add_argument(
        "--fold",
        type=int,
        help="Training fold index used in train mode.",
    )
    parser.add_argument(
        "--source-fold",
        type=int,
        help="Fold index of the trained Mask R-CNN model used in detect mode.",
    )
    parser.add_argument(
        "--target-fold",
        type=int,
        help="Fold index to run detection on and write bb_maps into.",
    )
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--checkpoint-policy",
        choices=("best", "last"),
        default=DETECT_CHECKPOINT_POLICY,
        help="Checkpoint selection policy used in detect mode when --checkpoint-path is not provided.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Explicit trained detector checkpoint to use in detect mode.",
    )
    parser.add_argument(
        "--train-fold",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-fold",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--detect-fold",
        type=int,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    preset = get_mask_rcnn_preset()
    if args.epochs is None:
        args.epochs = preset["epochs"]
    if args.batch_size is None:
        args.batch_size = preset["batch_size"]
    if args.learning_rate is None:
        args.learning_rate = preset["learning_rate"]

    if args.fold is not None and args.train_fold is not None and args.fold != args.train_fold:
        raise SystemExit("Conflicting values passed for --fold and --train-fold.")
    if args.source_fold is not None and args.model_fold is not None and args.source_fold != args.model_fold:
        raise SystemExit("Conflicting values passed for --source-fold and --model-fold.")
    if args.target_fold is not None and args.detect_fold is not None and args.target_fold != args.detect_fold:
        raise SystemExit("Conflicting values passed for --target-fold and --detect-fold.")

    args.train_fold = args.fold if args.fold is not None else args.train_fold
    args.model_fold = args.source_fold if args.source_fold is not None else args.model_fold
    args.detect_fold = args.target_fold if args.target_fold is not None else args.detect_fold
    return args


def _find_best_weights(model_dir):
    best_files = list(Path(model_dir).rglob("mask_rcnn_teeth_best.h5"))
    if not best_files:
        return None
    return sorted(best_files, key=lambda p: p.stat().st_mtime)[-1]


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    project_root = Path(__file__).resolve().parents[1]
    mask_rcnn_root = Path(
        os.environ.get("MASK_RCNN_ROOT", str(project_root / "src" / "mrcnn_tf2"))
    ).resolve()
    if not mask_rcnn_root.exists():
        raise SystemExit(f"MASK_RCNN_ROOT not found: {mask_rcnn_root}")
    if str(mask_rcnn_root) not in sys.path:
        sys.path.insert(0, str(mask_rcnn_root))

    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "1")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    # Ensure pip-installed CUDA libs are discoverable at runtime.
    import site
    import ctypes
    from glob import glob

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
    import tensorflow as tf

    tf.random.set_seed(SEED)

    mapping = load_class_map(CLASS_MAP_PATH)
    num_classes = max(mapping) + 1 if mapping else 1

    def _clear_tf_memory():
        # Clear TF graphs and cached tensors between runs to release memory.
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            try:
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass
        except Exception:
            pass
        import gc

        gc.collect()

    class TeethConfig(Config):
        NAME = "teeth"
        IMAGES_PER_GPU = args.batch_size
        GPU_COUNT = 1
        NUM_CLASSES = num_classes
        STEPS_PER_EPOCH = 1000
        DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE
        LEARNING_RATE = args.learning_rate
        LEARNING_MOMENTUM = 0.9
        BACKBONE = "resnet101"
        # Keep 512x1024 without square padding to reduce memory overhead.
        IMAGE_MIN_DIM = DEFAULT_IMAGE_MIN_DIM
        IMAGE_MAX_DIM = DEFAULT_IMAGE_MAX_DIM
        IMAGE_RESIZE_MODE = "none"
        USE_MINI_MASK = False

    class TeethDataset(utils.Dataset):
        def load_teeth(self, splits_dir, subset, class_map_path):
            mapping = load_class_map(class_map_path)
            class_ids = [cid for cid in sorted(mapping) if cid != 0]
            for class_id in class_ids:
                self.add_class("teeth", class_id, mapping[class_id])
            self._class_ids = class_ids

            images_dir = Path(splits_dir) / subset / "img"
            masks_dir = Path(splits_dir) / subset / "masks_semantic"
            for img_path, mask_path in list_image_mask_pairs(images_dir, masks_dir):
                image = Image.open(img_path)
                width, height = image.size
                # Stored dimensions should reflect the resized output.
                width = FIXED_IMAGE_WIDTH
                height = FIXED_IMAGE_HEIGHT
                self.add_image(
                    "teeth",
                    image_id=img_path.stem,
                    path=str(img_path),
                    mask_path=str(mask_path),
                    width=width,
                    height=height,
                )

        def load_image(self, image_id):
            info = self.image_info[image_id]
            image = Image.open(info["path"])
            image = image.resize(
                (FIXED_IMAGE_WIDTH, FIXED_IMAGE_HEIGHT), resample=Image.BILINEAR
            )
            image = np.array(image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            return image

        def load_mask(self, image_id):
            info = self.image_info[image_id]
            mask = Image.open(info["mask_path"])
            mask = mask.resize(
                (FIXED_IMAGE_WIDTH, FIXED_IMAGE_HEIGHT), resample=Image.NEAREST
            )
            mask = np.array(mask)
            if mask.ndim == 3:
                mask = mask[..., 0]

            instance_masks = []
            class_ids = []
            for class_id in self._class_ids:
                binary = mask == class_id
                if np.any(binary):
                    instance_masks.append(binary)
                    class_ids.append(class_id)

            if not instance_masks:
                height, width = mask.shape[:2]
                return np.zeros((height, width, 0), dtype=bool), np.array([], dtype=np.int32)

            masks = np.stack(instance_masks, axis=-1).astype(np.uint8)
            return masks, np.array(class_ids, dtype=np.int32)

        def image_reference(self, image_id):
            return self.image_info[image_id]["path"]

    config = TeethConfig()

    def _is_legacy_h5(path):
        return str(path).endswith((".h5", ".hdf5")) and not str(path).endswith(".weights.h5")

    def _load_weights(model):
        if WEIGHTS.lower() == "coco":
            weights_path = LOGS_DIR / "mask_rcnn_coco.h5"
            if not weights_path.exists():
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                utils.download_trained_weights(str(weights_path))
        elif WEIGHTS.lower() == "imagenet":
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = Path(WEIGHTS)
        model.load_weights(
            str(weights_path),
            by_name=_is_legacy_h5(weights_path),
            exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"],
        )
        return weights_path

    def _train_on_split(split_root, model_dir):
        train_pairs = list_image_mask_pairs(
            Path(split_root) / TRAIN_SUBSET / "img",
            Path(split_root) / TRAIN_SUBSET / "masks_semantic",
        )
        val_pairs = list_image_mask_pairs(
            Path(split_root) / VAL_SUBSET / "img",
            Path(split_root) / VAL_SUBSET / "masks_semantic",
        )
        validate_disjoint_pair_sets({"train": train_pairs, "val": val_pairs})
        summarize_mask_paths([mask for _, mask in train_pairs], num_classes, subset_name="train")
        summarize_mask_paths([mask for _, mask in val_pairs], num_classes, subset_name="val")

        dataset_train = TeethDataset()
        dataset_train.load_teeth(split_root, TRAIN_SUBSET, CLASS_MAP_PATH)
        dataset_train.prepare()

        dataset_val = TeethDataset()
        dataset_val.load_teeth(split_root, VAL_SUBSET, CLASS_MAP_PATH)
        dataset_val.prepare()

        if len(dataset_train.image_ids) > 0:
            config.STEPS_PER_EPOCH = max(1, len(dataset_train.image_ids) // config.IMAGES_PER_GPU)
            sample_image = dataset_train.load_image(dataset_train.image_ids[0])
            resized, _, _, _, _ = utils.resize_image(
                sample_image,
                min_dim=config.IMAGE_MIN_DIM,
                max_dim=config.IMAGE_MAX_DIM,
                min_scale=config.IMAGE_MIN_SCALE,
                mode=config.IMAGE_RESIZE_MODE,
            )
            config.IMAGE_SHAPE = np.array(
                [resized.shape[0], resized.shape[1], config.IMAGE_CHANNEL_COUNT]
            )
        if len(dataset_val.image_ids) > 0:
            config.VALIDATION_STEPS = max(
                1, len(dataset_val.image_ids) // config.IMAGES_PER_GPU
            )

        model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_dir)
        _load_weights(model)

        original_fit = model.keras_model.fit
        best_path = Path(model.log_dir) / f"mask_rcnn_{config.NAME.lower()}_best.h5"
        class BestOnlyCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, checkpoint_path, best_path):
                self.checkpoint_path = checkpoint_path
                self.best_path = Path(best_path)
                self.best_loss = None

            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                val_loss = logs.get("val_loss")
                epoch_path = Path(self.checkpoint_path.format(epoch=epoch + 1))
                if not epoch_path.exists():
                    return
                if val_loss is None or self.best_loss is None or val_loss < self.best_loss:
                    if val_loss is not None:
                        self.best_loss = float(val_loss)
                    try:
                        shutil.copy2(epoch_path, self.best_path)
                    except Exception:
                        pass
                try:
                    epoch_path.unlink()
                except Exception:
                    pass

        best_callback = BestOnlyCheckpoint(model.checkpoint_path, best_path)

        def _fit(*fit_args, **fit_kwargs):
            fit_kwargs["max_queue_size"] = min(
                fit_kwargs.get("max_queue_size", FIT_MAX_QUEUE_SIZE), FIT_MAX_QUEUE_SIZE
            )
            fit_kwargs["workers"] = 0
            fit_kwargs["use_multiprocessing"] = False
            return original_fit(*fit_args, **fit_kwargs)

        model.keras_model.fit = _fit
        try:
            print(f"Training all layers for {args.epochs} epochs...")
            model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=args.epochs,
                layers="all",
                custom_callbacks=[best_callback],
            )
        finally:
            model.keras_model.fit = original_fit
        return dataset_val, Path(model.log_dir)

    def _compute_map(model, dataset):
        aps = []
        for image_id in dataset.image_ids:
            image = dataset.load_image(image_id)
            gt_mask, gt_class_ids = dataset.load_mask(image_id)
            if gt_mask.size == 0:
                continue
            gt_bbox = utils.extract_bboxes(gt_mask)
            results = model.detect([image], verbose=0)[0]
            if results["rois"].size == 0:
                aps.append(0.0)
                continue
            ap, _, _, _ = utils.compute_ap(
                gt_bbox,
                gt_class_ids,
                gt_mask,
                results["rois"],
                results["class_ids"],
                results["scores"],
                results["masks"],
                iou_threshold=MAP_IOU_THRESHOLD,
            )
            aps.append(ap)
        return float(np.mean(aps)) if aps else 0.0

    def _latest_run_dir(base_dir):
        base_dir = Path(base_dir)
        candidates = [
            d for d in base_dir.iterdir()
            if d.is_dir() and d.name.startswith(config.NAME.lower())
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda d: d.name)[-1]

    def _best_checkpoint_from_events(run_dir):
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow.python.summary.summary_iterator import summary_iterator
        except Exception:
            return None

        run_dir = Path(run_dir)
        event_files = sorted(run_dir.glob("events.out.tfevents.*"))
        best_loss = None
        best_epoch = None

        for event_path in event_files:
            try:
                iterator = summary_iterator(str(event_path))
            except Exception:
                continue
            for ev in iterator:
                if not ev.summary:
                    continue
                for val in ev.summary.value:
                    tag = val.tag or ""
                    if "val_loss" not in tag:
                        continue
                    loss = float(val.simple_value)
                    epoch = max(1, int(ev.step))
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                        best_epoch = epoch

        if best_epoch is None:
            best_path = run_dir / f"mask_rcnn_{config.NAME.lower()}_best.h5"
            return (best_path, None, None) if best_path.exists() else None

        name = config.NAME
        for epoch in (best_epoch, best_epoch + 1):
            path = run_dir / f"mask_rcnn_{name}_{epoch:04d}.h5"
            if path.exists():
                return path, best_epoch, best_loss
        best_path = run_dir / f"mask_rcnn_{config.NAME.lower()}_best.h5"
        if best_path.exists():
            return best_path, best_epoch, best_loss
        return None

    def _make_inference_model(model_dir, weights_path=None):
        class TeethInferenceConfig(TeethConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        inference_config = TeethInferenceConfig()
        inference_config.NUM_CLASSES = num_classes

        model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=model_dir)
        if weights_path is None:
            weights_path = model.find_last()
            if not weights_path:
                raise SystemExit(f"Missing trained weights in {model_dir}")
        else:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise SystemExit(f"Missing trained weights file: {weights_path}")
        model.load_weights(str(weights_path), by_name=_is_legacy_h5(weights_path))
        return model, Path(weights_path)

    def _write_mask_rcnn_cv_summary(folds_root):
        per_fold = []
        for fold_path in sorted(LOGS_DIR.glob("cv_summary_fold_*.json")):
            try:
                payload = json.loads(fold_path.read_text())
                fold_idx = int(payload["fold"])
                map_score = float(payload["mAP@0.5"])
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            per_fold.append(
                {
                    "fold": fold_idx,
                    "mAP@0.5": map_score,
                    "checkpoint_policy": payload.get("checkpoint_policy", "best"),
                    "weights_path": payload.get("weights_path"),
                    "run_dir": payload.get("run_dir"),
                }
            )

        # Keep only one entry per fold index and sort by fold.
        unique = {}
        for row in per_fold:
            unique[row["fold"]] = row
        rows = [unique[idx] for idx in sorted(unique.keys())]

        expected_folds = sorted(
            int(path.name.split("_")[1]) for path in folds_root.glob("fold_*") if path.is_dir()
        )
        completed_folds = [row["fold"] for row in rows]
        missing_folds = [idx for idx in expected_folds if idx not in set(completed_folds)]

        map_scores = [row["mAP@0.5"] for row in rows]
        aggregate = {
            "num_completed_folds": len(rows),
            "mean_mAP@0.5": float(np.mean(map_scores)) if map_scores else None,
            "std_mAP@0.5": float(np.std(map_scores)) if len(map_scores) > 1 else 0.0,
            "expected_folds": expected_folds,
            "missing_folds": missing_folds,
            "is_complete": len(missing_folds) == 0,
        }

        summary = {"folds": rows, "aggregate": aggregate}
        summary_path = LOGS_DIR / "cv_summary.json"
        write_json(summary_path, summary)
        return summary_path

    if args.mode == "train":
        folds_root = SPLITS_DIR / "folds"
        if not folds_root.exists():
            raise SystemExit(f"Missing folds directory: {folds_root}")
        if args.train_fold is None:
            raise SystemExit(
                "For MODE='train', pass --fold <index> to run a single fold."
            )
        fold_dir = folds_root / f"fold_{args.train_fold}"
        if not fold_dir.exists():
            raise SystemExit(f"Missing fold directory: {fold_dir}")
        fold_log_dir = LOGS_DIR / f"fold_{args.train_fold}"
        print(f"Training fold {args.train_fold} in {fold_dir}...")
        dataset_val = None
        inference_model = None
        try:
            dataset_val, run_dir = _train_on_split(fold_dir, fold_log_dir)
            best_info = _best_checkpoint_from_events(run_dir)
            if best_info is None:
                best_info = _best_checkpoint_from_events(_latest_run_dir(fold_log_dir) or run_dir)
            best_weights = None
            if best_info is not None:
                best_weights, best_epoch, best_loss = best_info
                print(
                    f"Using best checkpoint for mAP: {best_weights.name} "
                    f"(epoch {best_epoch}, val_loss={best_loss:.6f})"
                )
            inference_model, resolved_weights = _make_inference_model(
                fold_log_dir, weights_path=best_weights
            )
            map_score = _compute_map(inference_model, dataset_val)
            summary = {
                "fold": args.train_fold,
                "mAP@0.5": map_score,
                "checkpoint_policy": "best",
                "weights_path": str(resolved_weights),
                "run_dir": str(run_dir),
                "seed": SEED,
            }
            summary_path = LOGS_DIR / f"cv_summary_fold_{args.train_fold}.json"
            write_json(summary_path, summary)
            print(f"Fold {args.train_fold} mAP@0.5: {map_score:.4f}")
            print(f"Wrote fold summary to {summary_path}")
            cv_summary_path = _write_mask_rcnn_cv_summary(folds_root)
            print(f"Updated CV summary at {cv_summary_path}")
        finally:
            del dataset_val
            del inference_model
            _clear_tf_memory()
        return

    class TeethInferenceConfig(TeethConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = TeethInferenceConfig()
    inference_config.NUM_CLASSES = num_classes

    detect_input_root = SPLITS_DIR
    detect_export_root = MASK_RCNN_BB_MAPS_ROOT
    detect_subsets = [DETECT_SUBSET]
    if args.detect_fold is not None:
        folds_root = SPLITS_DIR / "folds"
        if not folds_root.exists():
            raise SystemExit(f"Missing folds directory: {folds_root}")
        detect_input_root = folds_root / f"fold_{args.detect_fold}"
        if not detect_input_root.exists():
            raise SystemExit(f"Missing fold directory: {detect_input_root}")
        detect_export_root = detect_export_root / "folds" / f"fold_{args.detect_fold}"
        detect_subsets = [TRAIN_SUBSET, VAL_SUBSET]

    detect_logs_dir = LOGS_DIR
    if args.model_fold is None and args.checkpoint_path is None:
        raise SystemExit(
            "Detect mode requires an explicit detector source. Pass --source-fold or "
            "--checkpoint-path."
        )
    if args.model_fold is not None:
        detect_logs_dir = LOGS_DIR / f"fold_{args.model_fold}"
    print(f"Using model_dir for detection: {detect_logs_dir}")
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=detect_logs_dir)
    if args.checkpoint_path is not None:
        weights_path = Path(args.checkpoint_path)
        checkpoint_selection = "explicit_path"
    elif args.checkpoint_policy == "best":
        weights_path = _find_best_weights(detect_logs_dir)
        checkpoint_selection = "best"
        if weights_path is None:
            raise SystemExit(f"Missing best detector checkpoint in {detect_logs_dir}")
    else:
        weights_path = model.find_last()
        checkpoint_selection = "last"
        if not weights_path:
            raise SystemExit(f"Missing trained weights in {detect_logs_dir}")
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise SystemExit(f"Missing weights file: {weights_path}")
    print(f"Using detector checkpoint ({checkpoint_selection}): {weights_path}")
    model.load_weights(str(weights_path), by_name=_is_legacy_h5(weights_path))

    for subset in detect_subsets:
        dataset = TeethDataset()
        dataset.load_teeth(detect_input_root, subset, CLASS_MAP_PATH)
        dataset.prepare()

        output_dir = detect_export_root / subset / "bb_maps"
        output_dir.mkdir(parents=True, exist_ok=True)
        nonzero_bb_maps = 0
        negative_images = 0
        negative_images_with_nonzero_priors = 0

        for image_id in dataset.image_ids:
            image = dataset.load_image(image_id)
            results = model.detect([image], verbose=0)[0]
            bb_map = build_bb_map_from_boxes(
                results["rois"],
                results["class_ids"],
                results["scores"],
                image.shape[0],
                image.shape[1],
                num_classes,
                DETECTION_MIN_CONFIDENCE,
            )
            has_prior = bool(np.any(bb_map))
            nonzero_bb_maps += int(has_prior)
            mask = np.array(Image.open(dataset.image_info[image_id]["mask_path"]))
            if mask.ndim == 3:
                mask = mask[..., 0]
            is_negative = int(mask.max() == 0)
            negative_images += is_negative
            negative_images_with_nonzero_priors += int(is_negative and has_prior)
            out_path = output_dir / f"{dataset.image_info[image_id]['id']}.npz"
            np.savez_compressed(out_path, bb=bb_map)
        write_json(
            output_dir / "_export_metadata.json",
            {
                "subset": subset,
                "splits_root": str(detect_input_root),
                "export_root": str(detect_export_root),
                "source_model_dir": str(detect_logs_dir),
                "source_fold": args.model_fold,
                "checkpoint_policy": args.checkpoint_policy,
                "checkpoint_selection": checkpoint_selection,
                "weights_path": str(weights_path),
                "class_map_path": str(CLASS_MAP_PATH),
                "num_classes": int(num_classes),
                "bb_map_shape": [FIXED_IMAGE_HEIGHT, FIXED_IMAGE_WIDTH, num_classes],
                "images": len(dataset.image_ids),
                "nonzero_bb_maps": nonzero_bb_maps,
                "negative_images": negative_images,
                "negative_images_with_nonzero_priors": negative_images_with_nonzero_priors,
                "detection_min_confidence": DETECTION_MIN_CONFIDENCE,
                "seed": SEED,
            },
        )

    _clear_tf_memory()
    print(f"Saved BB maps under {detect_export_root}.")


if __name__ == "__main__":
    main()
