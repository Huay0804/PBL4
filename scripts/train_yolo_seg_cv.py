"""Run 4-fold CV training for an Ultralytics YOLO segmentation baseline.

Mirrors train_segmentation_cv.py: one model per fold, checkpoints land under
``runs/cv/fold_<k>/<run_name>/`` so evaluate_yolo_seg.py (and the existing
fold-merge tooling) find them in the same place as the dense segmenters.

Examples::

    python scripts/train_yolo_seg_cv.py --model yolo11            # proven baseline
    python scripts/train_yolo_seg_cv.py --model yolo26 --fold 0   # latest, one fold
    python scripts/train_yolo_seg_cv.py --model yolo11 --size l   # bigger backbone

Each fold writes ``runs/cv/fold_<k>/<run_name>/weights/best.pt`` plus a
``fold_train_meta.json`` capturing the resolved config.
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_presets import (
    DEFAULT_YOLO_SEG_MODEL,
    YOLO_SEG_MODELS,
    get_yolo_seg_preset,
    resolve_yolo_seg_model,
)
from protocol_utils import write_json
from prepare_yolo_seg_data import YOLO_SEG_ROOT, main as prepare_main

OUTPUT_ROOT = Path("runs/cv")
FOLDS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_YOLO_SEG_MODEL,
        help=f"Model alias ({'/'.join(YOLO_SEG_MODELS)}) or a raw ultralytics "
             f"weights spec (e.g. yolo11m-seg.pt).",
    )
    parser.add_argument("--size", default=None, help="Backbone size n/s/m/l/x (default: preset).")
    parser.add_argument("--fold", type=int, choices=range(FOLDS), help="Train a single fold.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help="CUDA device(s), e.g. '0' or '0,1'.")
    parser.add_argument(
        "--skip-prepare", action="store_true",
        help="Assume data/yolo_seg/* views already exist.",
    )
    return parser.parse_args()


def _ensure_dataset():
    if not (YOLO_SEG_ROOT / "test" / "data.yaml").exists() and not any(
        (YOLO_SEG_ROOT).glob("fold_*/data.yaml")
    ):
        print("Dataset views missing — materializing with prepare_yolo_seg_data...")
        prepare_main()


def train_fold(fold_index, model_spec, preset, args):
    from ultralytics import YOLO

    data_yaml = YOLO_SEG_ROOT / f"fold_{fold_index}" / "data.yaml"
    if not data_yaml.exists():
        raise SystemExit(
            f"Missing {data_yaml}. Run scripts/prepare_yolo_seg_data.py first "
            f"(or drop --skip-prepare)."
        )

    project_dir = OUTPUT_ROOT / f"fold_{fold_index}"
    run_name = model_spec["run_name"]

    epochs = args.epochs or preset["epochs"]
    batch = args.batch or preset["batch"]
    imgsz = args.imgsz or preset["imgsz"]

    print(f"\n===== TRAIN fold {fold_index} | {model_spec['weights']} =====", flush=True)
    model = YOLO(model_spec["weights"])
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        rect=preset["rect"],
        lr0=preset["learning_rate"],
        patience=preset["patience"],
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        device=args.device,
        seed=13,
    )

    best = project_dir / run_name / "weights" / "best.pt"
    write_json(
        project_dir / run_name / "fold_train_meta.json",
        {
            "fold": fold_index,
            "model_alias": model_spec["alias"],
            "weights_init": model_spec["weights"],
            "best_checkpoint": str(best),
            "data_yaml": str(data_yaml),
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "rect": preset["rect"],
        },
    )
    print(f"fold {fold_index} done -> {best}")
    return best


def main():
    args = parse_args()
    preset = get_yolo_seg_preset()
    model_spec = resolve_yolo_seg_model(args.model, size=args.size)
    if not args.skip_prepare:
        _ensure_dataset()

    folds = [args.fold] if args.fold is not None else list(range(FOLDS))
    for fold_index in folds:
        train_fold(fold_index, model_spec, preset, args)
    print("\nAll requested folds trained.")


if __name__ == "__main__":
    main()
