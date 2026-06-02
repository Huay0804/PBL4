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
import shutil
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

    # Use an ABSOLUTE project path: some Ultralytics versions nest a relative
    # project= under their default runs/<task>/ dir (e.g. runs/segment/runs/cv/
    # fold_0/...). An absolute path is used verbatim. The save_dir relocation
    # below is kept as a safety net for versions that ignore project= entirely.
    project_dir = (OUTPUT_ROOT / f"fold_{fold_index}").resolve()
    run_name = model_spec["run_name"]

    epochs = args.epochs or preset["epochs"]
    # Batch precedence: explicit --batch > per-model override > preset default.
    # -1 means Ultralytics auto-batch (probe GPU, pick ~60% utilization).
    requested_batch = args.batch or model_spec.get("batch") or preset["batch"]
    auto_batch = requested_batch == -1
    imgsz = args.imgsz or preset["imgsz"]

    print(f"\n===== TRAIN fold {fold_index} | {model_spec['weights']} | "
          f"batch={'auto (-1)' if auto_batch else requested_batch} =====", flush=True)
    model = YOLO(model_spec["weights"])
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=requested_batch,
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

    trainer = getattr(model, "trainer", None)

    # Some Ultralytics builds ignore project=/name= and save to their default
    # runs/<task>/ dir. The trainer's save_dir is ground truth — relocate it into
    # the canonical CV layout so eval and the results zip find the weights where
    # they expect them, regardless of the version's path handling.
    target_dir = project_dir / run_name
    actual_dir = Path(getattr(trainer, "save_dir", target_dir))
    if actual_dir.exists() and actual_dir.resolve() != target_dir.resolve():
        print(f"Ultralytics saved to {actual_dir}; relocating -> {target_dir}")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(actual_dir), str(target_dir))

    # After auto-batch, the trainer holds the size it actually chose. The
    # attribute name has shifted across Ultralytics versions, so probe both.
    resolved_batch = getattr(trainer, "batch_size", None)
    if resolved_batch in (None, -1) and trainer is not None:
        resolved_batch = getattr(getattr(trainer, "args", None), "batch", None)
    if resolved_batch in (None, -1):
        resolved_batch = requested_batch
    resolved_batch = int(resolved_batch)

    best = target_dir / "weights" / "best.pt"
    write_json(
        project_dir / run_name / "fold_train_meta.json",
        {
            "fold": fold_index,
            "model_alias": model_spec["alias"],
            "weights_init": model_spec["weights"],
            "best_checkpoint": str(best),
            "data_yaml": str(data_yaml),
            "epochs": epochs,
            "batch_requested": requested_batch,
            "batch_resolved": resolved_batch,
            "auto_batch": auto_batch,
            "imgsz": imgsz,
            "rect": preset["rect"],
        },
    )
    if auto_batch:
        print(f"fold {fold_index} done -> {best}  (auto-batch chose {resolved_batch})")
    else:
        print(f"fold {fold_index} done -> {best}  (batch {resolved_batch})")
    return resolved_batch


def main():
    args = parse_args()
    preset = get_yolo_seg_preset()
    model_spec = resolve_yolo_seg_model(args.model, size=args.size)
    if not args.skip_prepare:
        _ensure_dataset()

    folds = [args.fold] if args.fold is not None else list(range(FOLDS))
    resolved = {}
    for fold_index in folds:
        resolved[fold_index] = train_fold(fold_index, model_spec, preset, args)

    print("\nAll requested folds trained.")
    print(f"Batch size used per fold ({model_spec['alias']}):")
    for fold_index in folds:
        print(f"  fold {fold_index}: {resolved[fold_index]}")


if __name__ == "__main__":
    main()
