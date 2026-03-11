import argparse
import json
import sys
from pathlib import Path

import gc
import keras
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train import (
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    PREPROCESSING_255_BACKBONES,
    compute_confusion_matrix,
    infer_num_classes,
    list_pairs,
    load_class_map,
    make_dataset,
    report_group_metrics,
    report_per_class_metrics,
)
from segmentation_models.backbones import get_preprocessing
from helper_functions import (
    MeanIoUMetric,
    bce_dice_loss,
    ce_dice_loss,
    dice_coef,
    dice_coef_loss,
    iou_score,
    mean_iou,
    multiclass_dice_loss,
)
from project_presets import DEFAULT_SEGMENTATION_MODEL, get_segmentation_preset


SPLITS_DIR = Path("data/splits")
CLASS_MAP_PATH = Path("data/splits/class_map.txt")
RUNS_DIR = Path("runs")
INCLUDE_BACKGROUND = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved segmentation model on the fixed test set using project presets."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SEGMENTATION_MODEL,
        choices=["unet", "mod_unet", "nestnet", "linknet", "fpn", "icpr_unet", "icpr_munet"],
    )
    parser.add_argument("--backbone", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--loss",
        choices=["ce_dice", "bce_dice", "dice"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Override model run directory (expects best.keras/last.keras inside).",
    )
    parser.add_argument(
        "--cv-fold",
        type=int,
        choices=range(4),
        help="Evaluate a CV fold run (runs/cv/fold_<n>/<model-backbone>/).",
    )
    args = parser.parse_args()
    preset = get_segmentation_preset(args.model)
    if args.backbone is None:
        args.backbone = preset["backbone"]
    if args.batch_size is None:
        args.batch_size = preset["batch_size"]
    if args.loss is None:
        args.loss = preset["loss"]
    return args


def _backbone_passed(argv):
    return any(arg == "--backbone" or arg.startswith("--backbone=") for arg in argv)


def _summarize_overall(iou, dice, support, include_background):
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
        weighted_iou = 0.0
        weighted_dice = 0.0

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


def _select_loss(num_classes, loss_name):
    if num_classes == 1:
        if loss_name == "dice":
            return dice_coef_loss
        if loss_name == "bce_dice":
            return bce_dice_loss
        return ce_dice_loss
    if loss_name == "dice":
        return multiclass_dice_loss
    return ce_dice_loss


def _select_metrics(num_classes):
    if num_classes == 1:
        return [dice_coef, mean_iou, iou_score]
    return [
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
        MeanIoUMetric(num_classes=num_classes, name="mean_iou"),
    ]


def main():
    args = parse_args()
    keras.backend.clear_session()
    gc.collect()
    if args.model in ("icpr_unet", "icpr_munet") and _backbone_passed(sys.argv[1:]):
        raise SystemExit("--backbone is not supported for icpr_unet/icpr_munet.")
    run_name = f"{args.model}-{args.backbone}"
    if args.run_dir:
        run_dir = args.run_dir
    elif args.cv_fold is not None:
        run_dir = RUNS_DIR / "cv" / f"fold_{args.cv_fold}" / run_name
    else:
        run_dir = RUNS_DIR / run_name
    model_path = run_dir / "best.keras"
    if not model_path.exists():
        model_path = run_dir / "last.keras"
    if not model_path.exists():
        raise SystemExit(f"Model not found in {run_dir}")

    model = keras.models.load_model(model_path, compile=False)

    num_classes = infer_num_classes(CLASS_MAP_PATH) or 1
    if model.output_shape[-1] not in (None, 1) and model.output_shape[-1] != num_classes:
        print(
            f"warning: model output classes ({model.output_shape[-1]}) "
            f"do not match class_map ({num_classes})"
        )

    size = (DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH)
    use_backbone_preprocessing = args.model not in ("icpr_unet", "icpr_munet")
    preprocess_fn = get_preprocessing(args.backbone) if use_backbone_preprocessing else None
    scale_to_255 = use_backbone_preprocessing and args.backbone in PREPROCESSING_255_BACKBONES

    bb_maps_dir = None
    if len(model.inputs) > 1:
        bb_maps_dir = SPLITS_DIR / "test" / "bb_maps"
        if not bb_maps_dir.exists():
            raise SystemExit(f"BB maps directory not found: {bb_maps_dir}")

    bb_channels = None
    if len(model.inputs) > 1:
        inferred = model.inputs[1].shape[-1]
        bb_channels = inferred if inferred is not None else num_classes

    test_pairs = list_pairs(
        SPLITS_DIR / "test" / "img",
        SPLITS_DIR / "test" / "masks_semantic",
        bb_maps_dir=bb_maps_dir,
    )
    if not test_pairs:
        raise SystemExit("No test image/mask pairs found.")

    test_ds = make_dataset(
        test_pairs,
        size=size,
        num_classes=num_classes,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        preprocess_fn=preprocess_fn,
        scale_to_255=scale_to_255,
        bb_channels=bb_channels if len(model.inputs) > 1 else None,
    )

    loss = _select_loss(num_classes, args.loss)
    metrics = _select_metrics(num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=loss, metrics=metrics)

    report = model.evaluate(test_ds, verbose=1, return_dict=True)

    out_dir = model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(json.dumps(report, indent=2))
    print("Test metrics:", report)

    class_names = load_class_map(CLASS_MAP_PATH)
    class_names.setdefault(0, "background")

    # Free dataset memory before computing confusion matrix, then re-create
    # a small-batch dataset to reduce peak memory usage.
    del test_ds
    keras.backend.clear_session()
    gc.collect()

    cm_ds = make_dataset(
        test_pairs,
        size=size,
        num_classes=num_classes,
        batch_size=1,
        shuffle=False,
        augment=False,
        preprocess_fn=preprocess_fn,
        scale_to_255=scale_to_255,
        bb_channels=bb_channels if len(model.inputs) > 1 else None,
    )
    confusion = compute_confusion_matrix(cm_ds, model, num_classes)
    iou, dice, support = report_per_class_metrics("test", confusion, class_names, out_dir)
    report_group_metrics("test", iou, dice, support, out_dir, num_classes)

    summary = _summarize_overall(iou, dice, support, INCLUDE_BACKGROUND)
    (out_dir / "test_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote test summary to {out_dir / 'test_summary.json'}")

    keras.backend.clear_session()
    gc.collect()


if __name__ == "__main__":
    main()
