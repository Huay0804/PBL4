import argparse
from datetime import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

from project_presets import DEFAULT_SEGMENTATION_MODEL, get_segmentation_preset
from protocol_utils import load_json, merge_fold_rows, write_json


FOLDS_DIR = Path("data/splits/folds")
FOLDS = 4
OUTPUT_ROOT = Path("runs/cv")
INCLUDE_BACKGROUND = False
TRAIN_SCRIPT = Path("scripts/train.py")
PYTHON = sys.executable
YOLOX_BB_MAPS_FOLDS_DIR = Path(
    os.environ.get("PBL4_YOLOX_BB_MAPS_FOLDS_DIR", "data/bb_maps/yolox/folds")
)
MASK_RCNN_BB_MAPS_FOLDS_DIR = Path(
    os.environ.get("PBL4_MASK_RCNN_BB_MAPS_FOLDS_DIR", "data/bb_maps/mask_rcnn/folds")
)
BB_REQUIRED_MODELS = {"mod_nestnet", "icpr_munet"}
DEEP_SUPERVISION_HEAD_COUNT = 4
PROCESS_REFRESH_EXIT_CODE = 75


def _env_bool(name):
    value = os.environ.get(name)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "y")


def _env_int(name):
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run K-fold CV using the project training presets."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SEGMENTATION_MODEL,
        choices=[
            "mod_nestnet",
            "icpr_munet",
            "transunet",
            "icpr_unet",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--loss",
        choices=["ce_dice", "ce_dice_boundary", "bce_dice", "dice"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=["best", "last"],
        default="best",
        help="Checkpoint policy used when fold metrics are generated after training.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the fixed test split after each fold run.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=None,
        help="Use Keras mixed_float16 policy in each fold training run.",
    )
    parser.add_argument(
        "--bb-source",
        choices=("yolox", "mask_rcnn", "legacy_splits"),
        default=os.environ.get("PBL4_BB_SOURCE"),
        help=(
            "BB-map source for BB-gated models (mod_nestnet/icpr_munet). "
            "'yolox' uses data/bb_maps/yolox/folds, 'mask_rcnn' uses "
            "data/bb_maps/mask_rcnn/folds, 'legacy_splits' uses fold-local data/splits paths."
        ),
    )
    parser.add_argument(
        "--ds-inference",
        choices=["average", "last", "index"],
        default=None,
        help=(
            "Deep supervision inference output selection for fold post-training evaluation: "
            "'average' (mean of all DS heads), 'last' (final DS head), "
            "'index' (use --ds-output-index)."
        ),
    )
    parser.add_argument(
        "--ds-train-head",
        choices=["last", "all", "index"],
        default=os.environ.get("PBL4_DS_TRAIN_HEAD"),
        help=(
            "Training target for mod_nestnet deep-supervision heads. "
            "'last' trains only the final head, 'all' trains every head, "
            "'index' trains --ds-train-output-index only."
        ),
    )
    parser.add_argument(
        "--ds-output-index",
        type=int,
        default=None,
        help="Deep supervision output index when --ds-inference=index.",
    )
    parser.add_argument(
        "--ds-train-output-index",
        type=int,
        default=None,
        help=(
            "Zero-based UNet++ deep-supervision head index when --ds-train-head=index "
            "(0=shallowest, 3=final)."
        ),
    )
    parser.add_argument(
        "--process-restart-interval",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--fold", type=int, choices=range(FOLDS), help="Run a single fold (0-3).")
    args = parser.parse_args()
    preset = get_segmentation_preset(args.model)
    if args.batch_size is None:
        args.batch_size = preset["batch_size"]
    if args.epochs is None:
        args.epochs = preset["epochs"]
    if args.learning_rate is None:
        args.learning_rate = preset["learning_rate"]
    if args.loss is None:
        args.loss = preset["loss"]
    if args.bb_source is None:
        args.bb_source = preset.get("bb_source", "yolox")
    env_mixed_precision = _env_bool("PBL4_MIXED_PRECISION")
    if args.mixed_precision is None:
        args.mixed_precision = (
            env_mixed_precision
            if env_mixed_precision is not None
            else bool(preset.get("mixed_precision", False))
        )
    if args.ds_train_head is None:
        args.ds_train_head = preset.get("ds_train_head", "last")
    if args.ds_train_output_index is None:
        args.ds_train_output_index = preset.get("ds_train_output_index")
    if args.ds_inference is None:
        args.ds_inference = preset.get("ds_inference", "average")
    args.decoder_filters = preset.get("decoder_filters")
    args.rematerialization = preset.get("rematerialization")
    args.rematerialization_output_size_threshold = preset.get(
        "rematerialization_output_size_threshold",
        1048576,
    )
    args.early_stopping_patience = preset.get("early_stopping_patience")
    if args.process_restart_interval is None:
        args.process_restart_interval = _env_int("PBL4_PROCESS_RESTART_INTERVAL")
    if args.process_restart_interval is None:
        args.process_restart_interval = preset.get("process_restart_interval")
    if args.process_restart_interval is not None and args.process_restart_interval < 1:
        parser.error("--process-restart-interval must be a positive integer.")
    if args.ds_inference == "index" and args.ds_output_index is None:
        parser.error("--ds-output-index is required when --ds-inference=index.")
    if args.ds_inference != "index" and args.ds_output_index is not None:
        parser.error("--ds-output-index can only be used when --ds-inference=index.")
    if args.ds_train_head == "index" and args.ds_train_output_index is None:
        parser.error("--ds-train-output-index is required when --ds-train-head=index.")
    if args.ds_train_head != "index" and args.ds_train_output_index is not None:
        parser.error("--ds-train-output-index can only be used when --ds-train-head=index.")
    if (
        args.ds_train_output_index is not None
        and not 0 <= args.ds_train_output_index < DEEP_SUPERVISION_HEAD_COUNT
    ):
        parser.error(
            "--ds-train-output-index must be between 0 and "
            f"{DEEP_SUPERVISION_HEAD_COUNT - 1}."
        )
    return args


def resolve_deep_supervision_train_output_index(args):
    if args.model != "mod_nestnet" or args.ds_train_head == "all":
        return None
    if args.ds_train_head == "last":
        return DEEP_SUPERVISION_HEAD_COUNT - 1
    return args.ds_train_output_index


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


def _resolve_bb_maps_fold_dir(model_name, bb_source, fold_idx, fold_dir):
    if model_name not in BB_REQUIRED_MODELS:
        return fold_dir

    if bb_source == "legacy_splits":
        bb_fold_dir = fold_dir
    elif bb_source == "mask_rcnn":
        bb_fold_dir = MASK_RCNN_BB_MAPS_FOLDS_DIR / f"fold_{fold_idx}"
    else:
        bb_fold_dir = YOLOX_BB_MAPS_FOLDS_DIR / f"fold_{fold_idx}"

    if not bb_fold_dir.exists():
        raise SystemExit(
            f"Missing BB-maps fold directory for model '{model_name}' with source "
            f"'{bb_source}': {bb_fold_dir}"
        )
    for subset in ("train", "val"):
        subset_dir = bb_fold_dir / subset / "bb_maps"
        if not subset_dir.exists():
            raise SystemExit(f"Missing BB-maps subset directory: {subset_dir}")
    return bb_fold_dir


def _allocate_run_dir(run_root, run_name):
    stem = f"{run_name}{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    candidate = run_root / stem
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _find_resumable_run_dir(run_root, run_name):
    """Return the newest interrupted run dir for this fold, or None.

    A run dir is resumable when it still holds a ``.training_backup/`` (training
    was cut short between process refreshes) but has no ``last.keras`` (the final
    save was never reached). Reusing it lets BackupAndRestore continue the fold
    in place instead of starting a fresh timestamped dir on every re-run, which
    is what made checkpoints pile up.
    """
    if not run_root.exists():
        return None
    candidates = []
    for child in run_root.iterdir():
        if not child.is_dir() or not child.name.startswith(run_name):
            continue
        if (child / "last.keras").exists():
            continue  # fold already completed
        if (child / ".training_backup").is_dir():
            candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_run_dir(run_root, run_name):
    resumable = _find_resumable_run_dir(run_root, run_name)
    if resumable is not None:
        print(f"Resuming interrupted run dir: {resumable}")
        return resumable
    return _allocate_run_dir(run_root, run_name)


def main():
    args = parse_args()
    output_root = OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = args.model

    fold_metrics = []

    folds_to_run = [args.fold] if args.fold is not None else range(FOLDS)

    for i in folds_to_run:
        fold_dir = FOLDS_DIR / f"fold_{i}"
        if not fold_dir.exists():
            raise SystemExit(f"Missing fold directory: {fold_dir}")

        out_dir = output_root / f"fold_{i}"
        run_root = out_dir / run_name
        run_dir = _resolve_run_dir(run_root, run_name)
        cmd = [
            PYTHON,
            str(TRAIN_SCRIPT),
            "--model",
            args.model,
        ]
        cmd += [
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--loss",
            args.loss,
            "--checkpoint-policy",
            args.checkpoint_policy,
            "--ds-train-head",
            args.ds_train_head,
            "--ds-inference",
            args.ds_inference,
            "--run-dir",
            str(run_dir),
        ]
        if args.ds_train_head == "index":
            cmd += ["--ds-train-output-index", str(args.ds_train_output_index)]
        if args.ds_inference == "index":
            cmd += ["--ds-output-index", str(args.ds_output_index)]
        if args.eval_test:
            cmd.append("--eval-test")
        if args.mixed_precision:
            cmd.append("--mixed-precision")

        env = os.environ.copy()
        env["PBL4_SPLITS_DIR"] = str(fold_dir)
        env["PBL4_BB_MAPS_DIR"] = str(
            _resolve_bb_maps_fold_dir(args.model, args.bb_source, i, fold_dir)
        )
        env["PBL4_CLASS_MAP_PATH"] = str(FOLDS_DIR.parent / "class_map.txt")
        env["PBL4_OUTPUT_DIR"] = str(out_dir)
        if args.process_restart_interval:
            env["PBL4_PROCESS_RESTART_INTERVAL"] = str(args.process_restart_interval)

        print(f"Running fold {i}: {' '.join(cmd)}")
        print(f"Using BB maps root: {env['PBL4_BB_MAPS_DIR']}")
        while True:
            result = subprocess.run(cmd, env=env)
            if result.returncode == PROCESS_REFRESH_EXIT_CODE:
                print(
                    "Restarting training process to release CUDA memory pool; "
                    f"run dir remains {run_dir}"
                )
                continue
            result.check_returncode()
            break

        per_class_path = run_dir / "per_class_metrics_val.json"
        if not per_class_path.exists():
            compatibility_path = out_dir / run_name / "per_class_metrics_val.json"
            if compatibility_path.exists():
                per_class_path = compatibility_path
        if per_class_path.exists():
            metrics = _metrics_from_file(per_class_path, INCLUDE_BACKGROUND)
            if metrics:
                metrics["fold"] = i
                metrics["run_dir"] = str(run_dir)
                fold_metrics.append(metrics)
        else:
            print(f"warning: missing {per_class_path}")

    summary_path = output_root / f"{run_name}_cv_summary.json"
    existing = load_json(summary_path, default={}) or {}
    merged_folds = merge_fold_rows(existing.get("folds", []), fold_metrics)
    completed_folds = [row["fold"] for row in merged_folds]
    missing_folds = [idx for idx in range(FOLDS) if idx not in set(completed_folds)]
    summary = {
        "run_name": run_name,
        "model": args.model,
        "encoder": "icpr",
        "bb_source": args.bb_source if args.model in BB_REQUIRED_MODELS else None,
        "deep_supervision_train_strategy": (
            args.ds_train_head if args.model == "mod_nestnet" else None
        ),
        "deep_supervision_train_output_index": resolve_deep_supervision_train_output_index(args),
        "deep_supervision_eval_strategy": args.ds_inference,
        "deep_supervision_eval_output_index": (
            args.ds_output_index if args.ds_inference == "index" else None
        ),
        "checkpoint_policy": args.checkpoint_policy,
        "mixed_precision": bool(args.mixed_precision),
        "decoder_filters": list(args.decoder_filters) if args.decoder_filters else None,
        "process_restart_interval": (
            int(args.process_restart_interval) if args.process_restart_interval else None
        ),
        "early_stopping_patience": (
            int(args.early_stopping_patience) if args.early_stopping_patience else None
        ),
        "rematerialization": args.rematerialization,
        "rematerialization_output_size_threshold": (
            int(args.rematerialization_output_size_threshold)
            if args.rematerialization_output_size_threshold is not None
            else None
        ),
        "evaluated_test_during_training": bool(args.eval_test),
        "folds": merged_folds,
        "aggregate": summarize_folds(merged_folds),
        "expected_folds": list(range(FOLDS)),
        "completed_folds": completed_folds,
        "missing_folds": missing_folds,
        "is_complete": len(missing_folds) == 0,
    }
    write_json(summary_path, summary)
    print(f"Wrote CV summary to {summary_path}")


if __name__ == "__main__":
    main()
