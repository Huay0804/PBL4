import argparse
import gc
import json
from pathlib import Path

from project_presets import get_yolox_preset
from protocol_utils import write_json
import train_yolox as yolox_runner


def parse_args():
    preset = get_yolox_preset()
    parser = argparse.ArgumentParser(
        description="Re-evaluate existing YOLOX checkpoints with class-wise bbox_mAP@0.5."
    )
    parser.add_argument("--fold", type=int, help="Fold index to evaluate. Omit with --all-folds.")
    parser.add_argument("--all-folds", action="store_true", help="Evaluate every fold under splits-root/folds.")
    parser.add_argument("--subset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--splits-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--class-map", type=Path, default=Path("data/splits/class_map.txt"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/yolox"))
    parser.add_argument("--experiment-name", default=preset["experiment_name"])
    parser.add_argument("--checkpoint-policy", choices=("best", "last"), default="best")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--input-height", type=int, default=preset["input_height"])
    parser.add_argument("--input-width", type=int, default=preset["input_width"])
    parser.add_argument("--conf-threshold", type=float, default=preset["conf_threshold"])
    parser.add_argument("--nms-threshold", type=float, default=preset["nms_threshold"])
    parser.add_argument(
        "--class-agnostic-nms",
        action="store_true",
        help="Use class-agnostic NMS. Default is class-aware NMS.",
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    if not args.all_folds and args.fold is None:
        raise SystemExit("Pass --fold <index> or --all-folds.")
    if args.checkpoint_path is not None and (args.all_folds or args.fold is None):
        raise SystemExit("--checkpoint-path can only be used with one explicit --fold.")
    return args


def discover_folds(splits_root):
    folds_root = Path(splits_root) / "folds"
    return sorted(
        int(path.name.split("_", 1)[1])
        for path in folds_root.glob("fold_*")
        if path.is_dir() and path.name.split("_", 1)[1].isdigit()
    )


def update_train_summary(run_dir, eval_summary, checkpoint_path, checkpoint_meta):
    if not run_dir:
        return
    summary_path = Path(run_dir) / "train_summary.json"
    if not summary_path.exists():
        return
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    payload["evaluation"] = eval_summary
    payload["checkpoint"] = str(checkpoint_path)
    payload["checkpoint_meta"] = checkpoint_meta
    write_json(summary_path, payload)


def evaluate_fold(args, fold, detector_class_ids):
    if args.subset == "test":
        pairs = yolox_runner.list_image_mask_pairs(
            args.splits_root / "test" / "img",
            args.splits_root / "test" / "masks_semantic",
        )
        target_fold = None
    else:
        subset_root = args.splits_root / "folds" / f"fold_{fold}" / args.subset
        pairs = yolox_runner.list_image_mask_pairs(
            subset_root / "img",
            subset_root / "masks_semantic",
        )
        target_fold = fold
    if not pairs:
        raise SystemExit(f"No image-mask pairs found for fold {fold} subset '{args.subset}'.")

    predictor, checkpoint_path, checkpoint_meta = yolox_runner._load_predictor(
        fold,
        args.experiment_name,
        len(detector_class_ids),
        args,
    )
    eval_summary = yolox_runner._evaluate_predictor_on_pairs(
        predictor,
        detector_class_ids,
        pairs,
        args.conf_threshold,
        args.subset,
    )
    eval_summary.update(
        {
            "mode": "eval",
            "source_fold": int(fold),
            "target_fold": target_fold,
            "experiment_name": args.experiment_name,
            "checkpoint": str(checkpoint_path),
            "checkpoint_meta": checkpoint_meta,
            "input_size": [args.input_height, args.input_width],
            "conf_threshold": args.conf_threshold,
            "nms_threshold": args.nms_threshold,
            "class_agnostic_nms": args.class_agnostic_nms,
        }
    )

    run_dir = checkpoint_meta.get("run_dir")
    update_train_summary(run_dir, eval_summary, checkpoint_path, checkpoint_meta)

    fold_summary = {
        "fold": int(fold),
        "bbox_mAP@0.5": float(eval_summary["score"]),
        "metric_name": eval_summary["metric_name"],
        "metric_note": eval_summary["metric_note"],
        "subset": eval_summary["subset"],
        "conf_threshold": args.conf_threshold,
        "nms_threshold": args.nms_threshold,
        "class_agnostic_nms": eval_summary["class_agnostic_nms"],
        "num_images": eval_summary["num_images"],
        "num_gt_instances": eval_summary["num_gt_instances"],
        "num_predictions": eval_summary["num_predictions"],
        "class_ap": eval_summary["class_ap"],
        "table1_style_counts": eval_summary["table1_style_counts"],
        "checkpoint_policy": args.checkpoint_policy,
        "weights_path": str(checkpoint_path),
        "run_dir": run_dir,
        "experiment_name": args.experiment_name,
    }
    summary_path = args.runs_dir / f"cv_summary_fold_{fold}.json"
    write_json(summary_path, fold_summary)

    if run_dir:
        out_name = (
            f"eval_bbox_{args.subset}.json"
            if target_fold is None
            else f"eval_bbox_target_fold_{target_fold}_{args.subset}.json"
        )
        write_json(Path(run_dir) / out_name, eval_summary)

    del predictor
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return fold_summary


def main():
    args = parse_args()
    yolox_runner.SPLITS_DIR = args.splits_root
    yolox_runner.CLASS_MAP_PATH = args.class_map
    yolox_runner.RUNS_DIR = args.runs_dir

    yolox_runner._ensure_cuda_runtime_paths()
    yolox_runner._ensure_yolox_available()
    yolox_runner._configure_yolox_imports()
    if args.device == "gpu":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available; rerun with --device cpu.")

    mapping = yolox_runner.load_class_map(args.class_map)
    if not mapping:
        raise SystemExit(f"Class map not found or empty: {args.class_map}")
    detector_class_ids = [class_id for class_id in sorted(mapping) if class_id != 0]

    expected_folds = discover_folds(args.splits_root)
    folds = expected_folds if args.all_folds else [int(args.fold)]
    for fold in folds:
        print(f"Evaluating YOLOX fold {fold} on {args.subset}...")
        summary = evaluate_fold(args, fold, detector_class_ids)
        print(f"Fold {fold} bbox_mAP@0.5: {summary['bbox_mAP@0.5']:.4f}")

    cv_summary_path = yolox_runner._write_yolox_cv_summary(args.splits_root / "folds")
    print(f"Wrote aggregate summary: {cv_summary_path}")


if __name__ == "__main__":
    main()
