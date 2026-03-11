import json
import argparse
from pathlib import Path
from statistics import mean, pstdev


RUNS_DIR = Path("runs/cv")
FOLDS = [f"fold_{i}" for i in range(4)]
MODELS = ["icpr_unet-resnet18", "icpr_munet-resnet18"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate ICPR-style metrics across CV folds."
    )
    p.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    p.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Model run dir names under each fold.",
    )
    p.add_argument(
        "--per-position-split",
        choices=("val", "test"),
        default="test",
        help="Split used for per-tooth-position aggregation.",
    )
    p.add_argument(
        "--tooth-type-split",
        choices=("val", "test"),
        default="test",
        help="Split used for tooth-type aggregation.",
    )
    p.add_argument(
        "--out-per-position",
        type=Path,
        default=Path("runs/cv/summary_per_position_test.json"),
    )
    p.add_argument(
        "--out-tooth-type",
        type=Path,
        default=Path("runs/cv/summary_tooth_type_test.json"),
    )
    return p.parse_args()


def _load_json(path):
    return json.loads(path.read_text())


def _aggregate_per_class(runs_dir, model_dir, split):
    per_class = {}
    for fold in FOLDS:
        path = runs_dir / fold / model_dir / f"per_class_metrics_{split}.json"
        if not path.exists():
            continue
        rows = _load_json(path)
        for row in rows:
            class_id = int(row["class_id"])
            if class_id == 0:
                continue  # skip background for per-tooth-position summary
            per_class.setdefault(class_id, {"class_name": row["class_name"], "iou": [], "dice": []})
            per_class[class_id]["iou"].append(float(row["iou"]))
            per_class[class_id]["dice"].append(float(row["dice"]))
    summary = []
    for class_id in sorted(per_class.keys()):
        entry = per_class[class_id]
        iou_vals = entry["iou"]
        dice_vals = entry["dice"]
        summary.append(
            {
                "class_id": class_id,
                "class_name": entry["class_name"],
                "iou_mean": mean(iou_vals) if iou_vals else 0.0,
                "iou_std": pstdev(iou_vals) if len(iou_vals) > 1 else 0.0,
                "dice_mean": mean(dice_vals) if dice_vals else 0.0,
                "dice_std": pstdev(dice_vals) if len(dice_vals) > 1 else 0.0,
                "folds": len(iou_vals),
            }
        )
    return summary


def _aggregate_tooth_type(runs_dir, model_dir, split):
    per_group = {}
    for fold in FOLDS:
        path = runs_dir / fold / model_dir / f"per_tooth_type_metrics_{split}.json"
        if not path.exists():
            continue
        rows = _load_json(path)
        for row in rows:
            group = row["group"]
            per_group.setdefault(
                group, {"class_ids": row.get("class_ids", []), "iou": [], "dice": []}
            )
            per_group[group]["iou"].append(float(row["iou_mean"]))
            per_group[group]["dice"].append(float(row["dice_mean"]))
    summary = []
    for group in sorted(per_group.keys()):
        entry = per_group[group]
        iou_vals = entry["iou"]
        dice_vals = entry["dice"]
        summary.append(
            {
                "group": group,
                "class_ids": entry["class_ids"],
                "iou_mean": mean(iou_vals) if iou_vals else 0.0,
                "iou_std": pstdev(iou_vals) if len(iou_vals) > 1 else 0.0,
                "dice_mean": mean(dice_vals) if dice_vals else 0.0,
                "dice_std": pstdev(dice_vals) if len(dice_vals) > 1 else 0.0,
                "folds": len(iou_vals),
            }
        )
    return summary


def main():
    args = parse_args()
    per_position = {}
    tooth_type = {}
    for model in args.models:
        per_position[model] = _aggregate_per_class(
            args.runs_dir, model, args.per_position_split
        )
        tooth_type[model] = _aggregate_tooth_type(
            args.runs_dir, model, args.tooth_type_split
        )

    args.out_per_position.parent.mkdir(parents=True, exist_ok=True)
    args.out_tooth_type.parent.mkdir(parents=True, exist_ok=True)

    args.out_per_position.write_text(
        json.dumps(per_position, indent=2)
    )
    args.out_tooth_type.write_text(
        json.dumps(tooth_type, indent=2)
    )
    print("Wrote:", args.out_per_position)
    print("Wrote:", args.out_tooth_type)


if __name__ == "__main__":
    main()
