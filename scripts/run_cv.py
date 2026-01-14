import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run K-fold CV by invoking train.py per fold.")
    parser.add_argument("--folds-dir", default="data/splits/folds")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--class-map", default="data/splits/class_map.txt")
    parser.add_argument("--output-root", default="runs/cv")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument("--train-script", default="scripts/train.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()
    return args, unknown


def _drop_arg(unknown, name):
    cleaned = []
    skip = False
    for idx, item in enumerate(unknown):
        if skip:
            skip = False
            continue
        if item == name:
            skip = True
            continue
        if item.startswith(f"{name}="):
            continue
        cleaned.append(item)
    return cleaned


def _find_arg(unknown, name, default):
    for idx, item in enumerate(unknown):
        if item == name and idx + 1 < len(unknown):
            return unknown[idx + 1]
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return default


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
    args, unknown = parse_args()
    folds_dir = Path(args.folds_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    unknown = _drop_arg(unknown, "--splits-dir")
    unknown = _drop_arg(unknown, "--output-dir")
    unknown = _drop_arg(unknown, "--class-map")
    unknown = [arg for arg in unknown if arg != "--"]

    model_name = _find_arg(unknown, "--model", "unet")
    backbone = _find_arg(unknown, "--backbone", "resnet18")
    run_name = f"{model_name}-{backbone}"

    fold_metrics = []

    for i in range(args.folds):
        fold_dir = folds_dir / f"fold_{i}"
        if not fold_dir.exists():
            raise SystemExit(f"Missing fold directory: {fold_dir}")

        out_dir = output_root / f"fold_{i}"
        cmd = [
            args.python,
            args.train_script,
            "--splits-dir",
            str(fold_dir),
            "--class-map",
            args.class_map,
            "--output-dir",
            str(out_dir),
        ] + unknown

        print(f"Running fold {i}: {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)

        per_class_path = out_dir / run_name / "per_class_metrics_val.json"
        if per_class_path.exists():
            metrics = _metrics_from_file(per_class_path, args.include_background)
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
