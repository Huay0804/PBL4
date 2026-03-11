import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from project_presets import DEFAULT_SEGMENTATION_MODEL, get_segmentation_preset


FOLDS_DIR = Path("data/splits/folds")
FOLDS = 4
OUTPUT_ROOT = Path("runs/cv")
INCLUDE_BACKGROUND = False
TRAIN_SCRIPT = Path("scripts/train.py")
PYTHON = sys.executable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run K-fold CV using the project training presets."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SEGMENTATION_MODEL,
        choices=["unet", "mod_unet", "nestnet", "linknet", "fpn", "icpr_unet", "icpr_munet"],
    )
    parser.add_argument("--backbone", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--loss",
        choices=["ce_dice", "bce_dice", "dice"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fold", type=int, choices=range(FOLDS), help="Run a single fold (0-3).")
    args = parser.parse_args()
    preset = get_segmentation_preset(args.model)
    if args.backbone is None:
        args.backbone = preset["backbone"]
    if args.batch_size is None:
        args.batch_size = preset["batch_size"]
    if args.epochs is None:
        args.epochs = preset["epochs"]
    if args.learning_rate is None:
        args.learning_rate = preset["learning_rate"]
    if args.loss is None:
        args.loss = preset["loss"]
    return args


def _backbone_passed(argv):
    return any(arg == "--backbone" or arg.startswith("--backbone=") for arg in argv)


def _metrics_from_file(path, include_background):
    data = json.loads(path.read_text())
    rows = [r for r in data if include_background or r["class_id"] != 0]
    if not rows:
        return None
    iou = [r["iou"] for r in rows]
    dice = [r["dice"] for r in rows]
    pixels = [r["pixels"] for r in rows]
    pixel_sum = sum(pixels)

    macro_iou = float(sum(iou) / len(iou))
    macro_dice = float(sum(dice) / len(dice))
    if pixel_sum > 0:
        weighted_iou = float(sum(v * w for v, w in zip(iou, pixels)) / pixel_sum)
        weighted_dice = float(sum(v * w for v, w in zip(dice, pixels)) / pixel_sum)
    else:
        weighted_iou = 0.0
        weighted_dice = 0.0

    worst_idx = min(range(len(iou)), key=lambda idx: iou[idx])
    worst_class = rows[worst_idx]

    return {
        "macro_iou": macro_iou,
        "macro_dice": macro_dice,
        "weighted_iou": weighted_iou,
        "weighted_dice": weighted_dice,
        "worst_class_id": int(worst_class["class_id"]),
        "worst_class_name": worst_class["class_name"],
        "worst_iou": float(worst_class["iou"]),
        "worst_dice": float(worst_class["dice"]),
    }


def summarize_folds(fold_metrics):
    keys = [
        "macro_iou",
        "macro_dice",
        "weighted_iou",
        "weighted_dice",
        "worst_iou",
        "worst_dice",
    ]
    summary = {}
    for key in keys:
        values = [m[key] for m in fold_metrics if key in m]
        if not values:
            continue
        summary[f"mean_{key}"] = float(sum(values) / len(values))
    return summary


def main():
    args = parse_args()
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    if args.model in ("icpr_unet", "icpr_munet") and _backbone_passed(sys.argv[1:]):
        raise SystemExit("--backbone is not supported for icpr_unet/icpr_munet.")

    run_name = f"{args.model}-{args.backbone}"

    fold_metrics = []

    folds_to_run = [args.fold] if args.fold is not None else range(FOLDS)

    for i in folds_to_run:
        fold_dir = FOLDS_DIR / f"fold_{i}"
        if not fold_dir.exists():
            raise SystemExit(f"Missing fold directory: {fold_dir}")

        out_dir = output_root / f"fold_{i}"
        cmd = [
            PYTHON,
            str(TRAIN_SCRIPT),
            "--model",
            args.model,
        ]
        if args.model not in ("icpr_unet", "icpr_munet"):
            cmd += ["--backbone", args.backbone]
        cmd += [
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--loss",
            args.loss,
        ]

        env = os.environ.copy()
        env["PBL4_SPLITS_DIR"] = str(fold_dir)
        env["PBL4_BB_MAPS_DIR"] = str(fold_dir)
        env["PBL4_OUTPUT_DIR"] = str(out_dir)

        print(f"Running fold {i}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, env=env)

        per_class_path = out_dir / run_name / "per_class_metrics_val.json"
        if per_class_path.exists():
            metrics = _metrics_from_file(per_class_path, INCLUDE_BACKGROUND)
            if metrics:
                metrics["fold"] = i
                fold_metrics.append(metrics)
        else:
            print(f"warning: missing {per_class_path}")

    summary = {
        "folds": fold_metrics,
        "aggregate": summarize_folds(fold_metrics),
    }
    summary_path = output_root / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote CV summary to {summary_path}")


if __name__ == "__main__":
    main()
