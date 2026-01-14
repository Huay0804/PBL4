"""
Adapter script for the Matterport Mask R-CNN repo.
Requires the Mask_RCNN repo and its dependencies (scikit-image, scipy).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


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
    parser = argparse.ArgumentParser(description="Train or run Mask R-CNN on the teeth dataset.")
    parser.add_argument("command", choices=["train", "detect"])
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--class-map", default="data/splits/class_map.txt")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--logs", default="runs/mask_rcnn")
    parser.add_argument("--subset", default="train")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--images-per-gpu", type=int, default=1)
    parser.add_argument("--min-score", type=float, default=0.05)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    mask_rcnn_dir = os.environ.get("MASK_RCNN_DIR", "/home/huay/Mask_RCNN-master")
    mask_rcnn_dir = Path(mask_rcnn_dir)
    sys.path.insert(0, str(mask_rcnn_dir))

    from mrcnn.config import Config
    from mrcnn import model as modellib
    from mrcnn import utils

    class TeethConfig(Config):
        NAME = "teeth"
        IMAGES_PER_GPU = args.images_per_gpu
        NUM_CLASSES = 1
        STEPS_PER_EPOCH = 100
        DETECTION_MIN_CONFIDENCE = 0.05

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
            image = np.array(image)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            return image

        def load_mask(self, image_id):
            info = self.image_info[image_id]
            mask = np.array(Image.open(info["mask_path"]))
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

            masks = np.stack(instance_masks, axis=-1).astype(bool)
            return masks, np.array(class_ids, dtype=np.int32)

        def image_reference(self, image_id):
            return self.image_info[image_id]["path"]

    mapping = load_class_map(args.class_map)
    num_classes = max(mapping) + 1 if mapping else 1

    config = TeethConfig()
    config.NUM_CLASSES = num_classes

    if args.command == "train":
        dataset_train = TeethDataset()
        dataset_train.load_teeth(args.splits_dir, "train", args.class_map)
        dataset_train.prepare()

        dataset_val = TeethDataset()
        dataset_val.load_teeth(args.splits_dir, "val", args.class_map)
        dataset_val.prepare()

        if len(dataset_train.image_ids) > 0:
            config.STEPS_PER_EPOCH = max(1, len(dataset_train.image_ids) // config.IMAGES_PER_GPU)

        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
        if args.weights.lower() == "coco":
            weights_path = mask_rcnn_dir / "mask_rcnn_coco.h5"
        elif args.weights.lower() == "imagenet":
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = args.weights
        model.load_weights(str(weights_path), by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        model.train(
            dataset_train,
            dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=args.epochs,
            layers="all",
        )
        return

    class TeethInferenceConfig(TeethConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    inference_config = TeethInferenceConfig()
    inference_config.NUM_CLASSES = num_classes

    dataset = TeethDataset()
    dataset.load_teeth(args.splits_dir, args.subset, args.class_map)
    dataset.prepare()

    model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=args.logs)
    weights_path = args.weights
    if args.weights.lower() == "last":
        weights_path = model.find_last()
    model.load_weights(str(weights_path), by_name=True)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.splits_dir) / args.subset / "bb_maps"
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
            args.min_score,
        )
        out_path = output_dir / f"{dataset.image_info[image_id]['id']}.npz"
        np.savez_compressed(out_path, bb=bb_map)

    print(f"Saved BB maps to {output_dir}.")


if __name__ == "__main__":
    main()
