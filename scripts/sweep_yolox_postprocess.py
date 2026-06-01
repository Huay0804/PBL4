import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from project_presets import get_yolox_preset
from protocol_utils import write_json
import train_yolox as yolox_runner


def _parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args():
    preset = get_yolox_preset()
    parser = argparse.ArgumentParser(
        description="Sweep YOLOX confidence/NMS post-processing without retraining."
    )
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--subset", choices=("train", "val", "test"), default="val")
    parser.add_argument("--confs", default="0.01,0.03,0.05,0.08,0.10")
    parser.add_argument("--nms-thresholds", default="0.45,0.55,0.65,0.75")
    parser.add_argument(
        "--nms-modes",
        choices=("class_aware", "class_agnostic", "both"),
        default="both",
    )
    parser.add_argument("--splits-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--class-map", type=Path, default=Path("data/splits/class_map.txt"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/yolox"))
    parser.add_argument("--experiment-name", default=preset["experiment_name"])
    parser.add_argument("--checkpoint-policy", choices=("best", "last"), default="best")
    parser.add_argument("--input-height", type=int, default=preset["input_height"])
    parser.add_argument("--input-width", type=int, default=preset["input_width"])
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/yolox/postprocess_sweep.json"))
    return parser.parse_args()


def _pairs_for_fold(args, fold):
    if args.subset == "test":
        return yolox_runner.list_image_mask_pairs(
            args.splits_root / "test" / "img",
            args.splits_root / "test" / "masks_semantic",
        )
    subset_root = args.splits_root / "folds" / f"fold_{fold}" / args.subset
    return yolox_runner.list_image_mask_pairs(
        subset_root / "img",
        subset_root / "masks_semantic",
    )


def _nms_modes(args):
    if args.nms_modes == "both":
        return [False, True]
    return [args.nms_modes == "class_agnostic"]


def _mean_row(rows):
    scores = [row["bbox_mAP@0.5"] for row in rows]
    return {
        "mean_bbox_mAP@0.5": float(np.mean(scores)) if scores else None,
        "std_bbox_mAP@0.5": float(np.std(scores)) if len(scores) > 1 else 0.0,
        "folds": rows,
    }


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

    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    confs = _parse_float_list(args.confs)
    nms_thresholds = _parse_float_list(args.nms_thresholds)
    nms_modes = _nms_modes(args)

    rows_by_setting = {}
    checkpoint_rows = {}
    for fold in folds:
        pairs = _pairs_for_fold(args, fold)
        if not pairs:
            raise SystemExit(f"No image-mask pairs found for fold {fold} subset '{args.subset}'.")

        load_args = SimpleNamespace(
            input_height=args.input_height,
            input_width=args.input_width,
            conf_threshold=min(confs),
            nms_threshold=max(nms_thresholds),
            checkpoint_policy=args.checkpoint_policy,
            checkpoint_path=None,
            device=args.device,
            fp16=args.fp16,
            class_agnostic_nms=False,
        )
        predictor, checkpoint_path, checkpoint_meta = yolox_runner._load_predictor(
            fold,
            args.experiment_name,
            len(detector_class_ids),
            load_args,
        )
        checkpoint_rows[str(fold)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_meta": checkpoint_meta,
        }

        for class_agnostic in nms_modes:
            predictor.class_agnostic_nms = class_agnostic
            for nms_threshold in nms_thresholds:
                predictor.nmsthre = nms_threshold
                for conf_threshold in confs:
                    predictor.confthre = conf_threshold
                    result = yolox_runner._evaluate_predictor_on_pairs(
                        predictor,
                        detector_class_ids,
                        pairs,
                        conf_threshold,
                        args.subset,
                    )
                    key = (
                        f"{'class_agnostic' if class_agnostic else 'class_aware'}"
                        f"|conf={conf_threshold:g}|nms={nms_threshold:g}"
                    )
                    rows_by_setting.setdefault(
                        key,
                        {
                            "class_agnostic_nms": class_agnostic,
                            "conf_threshold": conf_threshold,
                            "nms_threshold": nms_threshold,
                            "folds": [],
                        },
                    )
                    counts = result["table1_style_counts"]
                    rows_by_setting[key]["folds"].append(
                        {
                            "fold": fold,
                            "bbox_mAP@0.5": float(result["score"]),
                            "num_predictions": int(result["num_predictions"]),
                            "correct": int(counts["detected_and_correctly_classified"]),
                            "misclassified": int(counts["miss_classified_detections"]),
                            "missed": int(counts["missed_detections"]),
                            "false_positive": int(counts["false_positive_detections"]),
                        }
                    )
                    print(
                        f"fold={fold} "
                        f"mode={'agnostic' if class_agnostic else 'aware'} "
                        f"conf={conf_threshold:g} nms={nms_threshold:g} "
                        f"bbox_mAP@0.5={result['score']:.4f}"
                    )

        del predictor
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    settings = []
    for key, row in rows_by_setting.items():
        aggregate = _mean_row(row["folds"])
        settings.append(
            {
                "setting": key,
                "class_agnostic_nms": row["class_agnostic_nms"],
                "conf_threshold": row["conf_threshold"],
                "nms_threshold": row["nms_threshold"],
                **aggregate,
            }
        )
    settings.sort(key=lambda row: row["mean_bbox_mAP@0.5"], reverse=True)

    payload = {
        "metric_name": "bbox_mAP@0.5",
        "subset": args.subset,
        "folds": folds,
        "checkpoints": checkpoint_rows,
        "settings": settings,
        "best": settings[0] if settings else None,
    }
    write_json(args.out, payload)
    if settings:
        best = settings[0]
        print(
            "Best setting: "
            f"{best['setting']} mean_bbox_mAP@0.5={best['mean_bbox_mAP@0.5']:.4f}"
        )
    print(f"Wrote sweep results to {args.out}")


if __name__ == "__main__":
    main()
