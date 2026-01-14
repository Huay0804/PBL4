import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import tensorflow as tf
import keras
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmentation_models import Unet, ModifiedUnet, Nestnet, Linknet, FPN
from segmentation_models.backbones import get_preprocessing
from helper_functions import (
    MeanIoUMetric,
    dice_coef,
    dice_coef_loss,
    bce_dice_loss,
    ce_dice_loss,
    multiclass_dice_loss,
    mean_iou,
    iou_score,
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_INPUT_HEIGHT = 512
DEFAULT_INPUT_WIDTH = 1024
DEFAULT_DECODER_BLOCK_TYPE = "upsampling"
PREPROCESSING_255_BACKBONES = {
    "vgg16",
    "vgg19",
    "densenet121",
    "densenet169",
    "densenet201",
    "inceptionv3",
    "inceptionresnetv2",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation models.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--class-map", default="data/splits/class_map.txt")
    parser.add_argument("--model", default="unet", choices=["unet", "mod_unet", "nestnet", "linknet", "fpn"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--loss", choices=["ce_dice", "bce_dice", "dice"], default="ce_dice")
    parser.add_argument("--class-weighting", choices=["none", "median"], default="none")
    parser.add_argument("--bb-maps-dir", default=None)
    parser.add_argument("--bb-channels", type=int, default=None)
    parser.add_argument("--output-dir", default="runs")
    return parser.parse_args()


def configure_tensorflow_memory():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def infer_num_classes(class_map_path):
    path = Path(class_map_path)
    if not path.exists():
        return None
    max_id = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            max_id = max(max_id, int(parts[-1]))
        except ValueError:
            continue
    if max_id == 0:
        return None
    return max_id + 1


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


def compute_class_weights(mask_paths, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in mask_paths:
        mask = np.array(Image.open(path))
        counts += np.bincount(mask.reshape(-1), minlength=num_classes)

    counts = counts.astype(np.float64)
    if num_classes > 1:
        non_bg = counts[1:]
        non_zero = non_bg[non_bg > 0]
        if non_zero.size == 0:
            return np.ones(num_classes, dtype=np.float32)
        median = np.median(non_zero)
    else:
        median = counts[0] if counts[0] > 0 else 1.0

    weights = median / np.maximum(counts, 1.0)
    return weights.astype(np.float32)


def make_weighted_losses(class_weights, num_classes):
    weights = tf.constant(class_weights, dtype=tf.float32)

    def weighted_sparse_ce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        w = tf.gather(weights, y_true)
        return tf.reduce_mean(ce * w)

    def weighted_dice(y_true, y_pred, smooth=1.0):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes, dtype=tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        axes = (0, 1, 2)
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        denom = tf.reduce_sum(y_true + y_pred, axis=axes)
        dice = (2.0 * intersection + smooth) / (denom + smooth)
        return 1.0 - tf.reduce_sum(dice * weights) / tf.reduce_sum(weights)

    def weighted_ce_dice(y_true, y_pred):
        return weighted_sparse_ce(y_true, y_pred) + weighted_dice(y_true, y_pred)

    return weighted_ce_dice, weighted_dice


def list_pairs(images_dir, masks_dir, bb_maps_dir=None, bb_ext=".npz"):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    masks = {p.stem: p for p in masks_dir.glob("*.png")}
    bb_maps = {}
    if bb_maps_dir is not None:
        bb_maps_dir = Path(bb_maps_dir)
        bb_maps = {p.stem: p for p in bb_maps_dir.glob(f"*{bb_ext}")}
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    pairs = []
    missing_bb = []
    for img in sorted(image_paths):
        mask = masks.get(img.stem)
        if mask is not None:
            if bb_maps_dir is None:
                pairs.append((str(img), str(mask)))
            else:
                bb_path = bb_maps.get(img.stem)
                if bb_path is None:
                    missing_bb.append(img.name)
                    continue
                pairs.append((str(img), str(mask), str(bb_path)))
    if bb_maps_dir is not None and missing_bb:
        sample = ", ".join(missing_bb[:5])
        raise SystemExit(f"Missing {len(missing_bb)} BB map files in {bb_maps_dir} (e.g. {sample})")
    return pairs


def _decode_image(path):
    data = tf.io.read_file(path)
    img = tf.image.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def _decode_mask(path):
    data = tf.io.read_file(path)
    mask = tf.image.decode_png(data, channels=1)
    return mask


def _load_bb_map_np(path):
    data = np.load(path.decode("utf-8"))
    if isinstance(data, np.lib.npyio.NpzFile):
        if "bb" in data:
            array = data["bb"]
        else:
            array = data[data.files[0]]
    else:
        array = data
    return array.astype(np.float32)


def load_bb_map(path, size, bb_channels):
    bb_map = tf.numpy_function(_load_bb_map_np, [path], tf.float32)
    bb_map.set_shape([None, None, bb_channels])
    if size is not None:
        bb_map = tf.image.resize(bb_map, size, method="nearest")
    bb_map = tf.cast(bb_map > 0.5, tf.float32)
    return bb_map


def load_pair(img_path, mask_path, size, num_classes, preprocess_fn, scale_to_255):
    image = _decode_image(img_path)
    mask = _decode_mask(mask_path)

    if size is not None:
        image = tf.image.resize(image, size, method="bilinear")
        mask = tf.image.resize(mask, size, method="nearest")

    if preprocess_fn is not None:
        if scale_to_255:
            image = image * 255.0
        image = preprocess_fn(image)

    if num_classes == 1:
        mask = tf.cast(mask > 0, tf.float32)
    else:
        mask = tf.cast(tf.squeeze(mask, axis=-1), tf.int32)

    return image, mask


def load_pair_with_bb(
    img_path,
    mask_path,
    bb_path,
    size,
    num_classes,
    preprocess_fn,
    scale_to_255,
    bb_channels,
):
    image, mask = load_pair(img_path, mask_path, size, num_classes, preprocess_fn, scale_to_255)
    bb_map = load_bb_map(bb_path, size, bb_channels)
    return (image, bb_map), mask



def _safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def compute_confusion_matrix(dataset, model, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, masks in dataset:
        preds = model.predict(images, verbose=0)
        if preds.shape[-1] == 1:
            pred_labels = (preds[..., 0] > 0.5).astype(np.int32)
            true_labels = masks.numpy().astype(np.int32)
        else:
            pred_labels = np.argmax(preds, axis=-1).astype(np.int32)
            true_labels = masks.numpy().astype(np.int32)
        conf = tf.math.confusion_matrix(
            true_labels.reshape(-1),
            pred_labels.reshape(-1),
            num_classes=num_classes,
        ).numpy()
        confusion += conf
    return confusion


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


def make_dataset(
    pairs,
    size,
    num_classes,
    batch_size,
    shuffle,
    augment,
    preprocess_fn,
    scale_to_255,
    bb_channels=None,
):
    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    if bb_channels is None:
        ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    else:
        bb_paths = [p[2] for p in pairs]
        ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, bb_paths))
    if shuffle:
        ds = ds.shuffle(len(pairs), reshuffle_each_iteration=True)
    if bb_channels is None:
        ds = ds.map(
            lambda x, y: load_pair(x, y, size, num_classes, preprocess_fn, scale_to_255),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda x, y, z: load_pair_with_bb(
                x, y, z, size, num_classes, preprocess_fn, scale_to_255, bb_channels
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(args, num_classes, input_shape, bb_channels):
    activation = "sigmoid" if num_classes == 1 else "softmax"
    classes = 1 if num_classes == 1 else num_classes

    if args.model == "unet":
        return Unet(
            backbone_name=args.backbone,
            input_shape=input_shape,
            encoder_weights=None,
            decoder_block_type=DEFAULT_DECODER_BLOCK_TYPE,
            classes=classes,
            activation=activation,
        )
    if args.model == "mod_unet":
        return ModifiedUnet(
            backbone_name=args.backbone,
            input_shape=input_shape,
            encoder_weights=None,
            decoder_block_type=DEFAULT_DECODER_BLOCK_TYPE,
            classes=classes,
            activation=activation,
            bb_channels=bb_channels,
        )
    if args.model == "nestnet":
        return Nestnet(
            backbone_name=args.backbone,
            input_shape=input_shape,
            encoder_weights=None,
            decoder_block_type=DEFAULT_DECODER_BLOCK_TYPE,
            classes=classes,
            activation=activation,
        )
    if args.model == "linknet":
        return Linknet(
            backbone_name=args.backbone,
            input_shape=input_shape,
            encoder_weights=None,
            classes=classes,
            activation=activation,
        )
    if args.model == "fpn":
        return FPN(
            backbone_name=args.backbone,
            input_shape=input_shape,
            encoder_weights=None,
            classes=classes,
            activation=activation,
        )
    raise ValueError(f"Unknown model {args.model}")


def main():
    args = parse_args()
    configure_tensorflow_memory()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    height = DEFAULT_INPUT_HEIGHT
    width = DEFAULT_INPUT_WIDTH

    input_shape = (height, width, 3)
    size = (height, width)

    num_classes = infer_num_classes(args.class_map) or 1
    preprocess_fn = get_preprocessing(args.backbone)
    scale_to_255 = args.backbone in PREPROCESSING_255_BACKBONES
    bb_maps_root = Path(args.bb_maps_dir) if args.bb_maps_dir else None
    if args.model == "mod_unet" and bb_maps_root is None:
        raise SystemExit("Modified U-Net requires --bb-maps-dir.")
    if args.model != "mod_unet" and bb_maps_root is not None:
        raise SystemExit("--bb-maps-dir is only supported with --model mod_unet.")
    bb_channels = args.bb_channels or num_classes
    if args.model != "mod_unet":
        bb_channels = None

    base = Path(args.splits_dir)
    train_pairs = list_pairs(
        base / "train" / "img",
        base / "train" / "masks_semantic",
        bb_maps_dir=bb_maps_root / "train" / "bb_maps" if bb_maps_root else None,
    )
    val_pairs = list_pairs(
        base / "val" / "img",
        base / "val" / "masks_semantic",
        bb_maps_dir=bb_maps_root / "val" / "bb_maps" if bb_maps_root else None,
    )
    test_pairs = list_pairs(
        base / "test" / "img",
        base / "test" / "masks_semantic",
        bb_maps_dir=bb_maps_root / "test" / "bb_maps" if bb_maps_root else None,
    )
    if not train_pairs:
        raise SystemExit("No training image/mask pairs found.")
    if not val_pairs:
        raise SystemExit("No validation image/mask pairs found.")

    class_weights = None
    if num_classes > 1 and args.class_weighting != "none":
        mask_paths = [pair[1] for pair in train_pairs]
        class_weights = compute_class_weights(mask_paths, num_classes)
        print(
            "Class weighting enabled:",
            f"min={class_weights.min():.4f}",
            f"max={class_weights.max():.4f}",
        )

    train_ds = make_dataset(
        train_pairs,
        size=size,
        num_classes=num_classes,
        batch_size=args.batch_size,
        shuffle=True,
        augment=False,
        preprocess_fn=preprocess_fn,
        scale_to_255=scale_to_255,
        bb_channels=bb_channels,
    )
    val_ds = make_dataset(
        val_pairs,
        size=size,
        num_classes=num_classes,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        preprocess_fn=preprocess_fn,
        scale_to_255=scale_to_255,
        bb_channels=bb_channels,
    )

    model = build_model(args, num_classes, input_shape, bb_channels)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    if num_classes == 1:
        if args.loss == "dice":
            loss = dice_coef_loss
        elif args.loss == "bce_dice":
            loss = bce_dice_loss
        else:
            loss = ce_dice_loss
        metrics = [dice_coef, mean_iou, iou_score]
    else:
        if class_weights is not None:
            weighted_ce_dice, weighted_dice = make_weighted_losses(class_weights, num_classes)
            loss = weighted_dice if args.loss == "dice" else weighted_ce_dice
        else:
            loss = ce_dice_loss if args.loss != "dice" else multiclass_dice_loss
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            MeanIoUMetric(num_classes=num_classes, name="mean_iou"),
        ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    run_name = f"{args.model}-{args.backbone}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "logs")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds if val_pairs else None,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(out_dir / "last.keras")

    if num_classes > 1:
        class_names = load_class_map(args.class_map)
        class_names.setdefault(0, "background")
        val_confusion = compute_confusion_matrix(val_ds, model, num_classes)
        val_iou, val_dice, val_support = report_per_class_metrics(
            "val", val_confusion, class_names, out_dir
        )
        report_group_metrics("val", val_iou, val_dice, val_support, out_dir, num_classes)

    if test_pairs:
        test_ds = make_dataset(
            test_pairs,
            size=size,
            num_classes=num_classes,
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
            preprocess_fn=preprocess_fn,
            scale_to_255=scale_to_255,
            bb_channels=bb_channels,
        )
        results = model.evaluate(test_ds, verbose=1)
        metric_names = model.metrics_names
        report = dict(zip(metric_names, results))
        print("Test metrics:", report)
        if num_classes > 1:
            test_confusion = compute_confusion_matrix(test_ds, model, num_classes)
            test_iou, test_dice, test_support = report_per_class_metrics(
                "test", test_confusion, class_names, out_dir
            )
            report_group_metrics("test", test_iou, test_dice, test_support, out_dir, num_classes)


if __name__ == "__main__":
    main()
