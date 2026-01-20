"""
Train a TF2-compatible Mask R-CNN implementation for teeth detection.
Edit MODE below to switch between training and BB-map export.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png")

SPLITS_DIR = Path(os.environ.get("PBL4_SPLITS_DIR", "data/splits"))
CLASS_MAP_PATH = Path("data/splits/class_map.txt")
LOGS_DIR = Path("runs/mask_rcnn")
WEIGHTS = "coco"
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
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on the teeth dataset.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--fold", type=int, help="Train/eval a single fold index.")
    return parser.parse_args()


def main():
    args = parse_args()
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
        STEPS_PER_EPOCH = 100
        DETECTION_MIN_CONFIDENCE = DETECTION_MIN_CONFIDENCE
        LEARNING_RATE = args.learning_rate
        LEARNING_MOMENTUM = 0.9
        BACKBONE = "resnet101"
        # Keep 512x1024 without square padding to reduce memory overhead.
        IMAGE_MIN_DIM = DEFAULT_IMAGE_MIN_DIM
        IMAGE_MAX_DIM = DEFAULT_IMAGE_MAX_DIM
        IMAGE_RESIZE_MODE = "none"
        # Tighten training ROI/anchor counts for lower memory use.
        RPN_TRAIN_ANCHORS_PER_IMAGE = 64
        TRAIN_ROIS_PER_IMAGE = 64
        POST_NMS_ROIS_TRAINING = 500
        POST_NMS_ROIS_INFERENCE = 500
        MAX_GT_INSTANCES = 32
        DETECTION_MAX_INSTANCES = 32
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

        def _fit(*fit_args, **fit_kwargs):
            fit_kwargs["max_queue_size"] = min(
                fit_kwargs.get("max_queue_size", FIT_MAX_QUEUE_SIZE), FIT_MAX_QUEUE_SIZE
            )
            fit_kwargs["workers"] = 0
            fit_kwargs["use_multiprocessing"] = False
            return original_fit(*fit_args, **fit_kwargs)

        model.keras_model.fit = _fit
        head_epochs = min(10, args.epochs)
        try:
            print(f"Training heads for {head_epochs} epochs...")
            model.train(
                dataset_train,
                dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=head_epochs,
                layers="heads",
            )
            if args.epochs > head_epochs:
                print(f"Training all layers until epoch {args.epochs}...")
                model.train(
                    dataset_train,
                    dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=args.epochs,
                    layers="all",
                )
        finally:
            model.keras_model.fit = original_fit
        return dataset_val

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

    def _make_inference_model(model_dir):
        class TeethInferenceConfig(TeethConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        inference_config = TeethInferenceConfig()
        inference_config.NUM_CLASSES = num_classes

        model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=model_dir)
        weights_path = model.find_last()
        if not weights_path:
            raise SystemExit(f"Missing trained weights in {model_dir}")
        model.load_weights(str(weights_path), by_name=_is_legacy_h5(weights_path))
        return model

    if MODE == "train":
        folds_root = SPLITS_DIR / "folds"
        if not folds_root.exists():
            raise SystemExit(f"Missing folds directory: {folds_root}")
        if args.fold is None:
            raise SystemExit("For MODE='train', pass --fold <index> to run a single fold.")
        fold_dir = folds_root / f"fold_{args.fold}"
        if not fold_dir.exists():
            raise SystemExit(f"Missing fold directory: {fold_dir}")
        fold_log_dir = LOGS_DIR / f"fold_{args.fold}"
        print(f"Training fold {args.fold} in {fold_dir}...")
        dataset_val = None
        inference_model = None
        try:
            dataset_val = _train_on_split(fold_dir, fold_log_dir)
            inference_model = _make_inference_model(fold_log_dir)
            map_score = _compute_map(inference_model, dataset_val)
            summary = {"fold": args.fold, "mAP@0.5": map_score}
            summary_path = LOGS_DIR / f"cv_summary_fold_{args.fold}.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            print(f"Fold {args.fold} mAP@0.5: {map_score:.4f}")
            print(f"Wrote fold summary to {summary_path}")
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

    dataset = TeethDataset()
    dataset.load_teeth(SPLITS_DIR, DETECT_SUBSET, CLASS_MAP_PATH)
    dataset.prepare()

    detect_logs_dir = LOGS_DIR
    if args.fold is not None:
        detect_logs_dir = LOGS_DIR / f"fold_{args.fold}"
    else:
        direct_runs = [
            d for d in LOGS_DIR.iterdir()
            if d.is_dir() and d.name.startswith(inference_config.NAME.lower())
        ]
        if not direct_runs:
            fold_dirs = sorted(LOGS_DIR.glob("fold_*"))
            for fold_dir in reversed(fold_dirs):
                has_run = any(
                    d.is_dir() and d.name.startswith(inference_config.NAME.lower())
                    for d in fold_dir.iterdir()
                )
                if has_run:
                    detect_logs_dir = fold_dir
                    break
    print(f"Using model_dir for detection: {detect_logs_dir}")
    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=detect_logs_dir)
    weights_path = WEIGHTS
    if WEIGHTS.lower() in ("coco", "imagenet"):
        weights_path = "last"
    if weights_path.lower() == "last":
        weights_path = model.find_last()
        if not weights_path:
            raise SystemExit(f"Missing trained weights in {detect_logs_dir}")
    else:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise SystemExit(f"Missing weights file: {weights_path}")
    model.load_weights(str(weights_path), by_name=_is_legacy_h5(weights_path))

    output_dir = SPLITS_DIR / DETECT_SUBSET / "bb_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        out_path = output_dir / f"{dataset.image_info[image_id]['id']}.npz"
        np.savez_compressed(out_path, bb=bb_map)

    _clear_tf_memory()
    print(f"Saved BB maps to {output_dir}.")


if __name__ == "__main__":
    main()
