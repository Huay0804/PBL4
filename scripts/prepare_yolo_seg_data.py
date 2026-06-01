"""Materialize fold-local Ultralytics YOLO-seg dataset views from the splits.

For each CV fold this writes::

    data/yolo_seg/fold_<k>/
        images/train/*.jpg   (symlinks into data/splits/folds/<k>/train/img)
        images/val/*.jpg
        labels/train/*.txt   (polygon labels derived from masks_semantic)
        labels/val/*.txt
        data.yaml            (Ultralytics dataset config)

The fixed test split is materialized once at ``data/yolo_seg/test/`` and wired
into every fold's data.yaml as the ``test:`` entry, so evaluation always scores
the same held-out images as the dense segmenters.

This is the YOLO-seg analogue of train_yolox.py's COCO materialization. It is
pure file I/O (no GPU / no ultralytics) so it can run anywhere, including the
local venv, before uploading the splits to Kaggle.
"""

import argparse
import os
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from project_presets import get_yolo_seg_preset
from protocol_utils import write_json
from yolo_seg_utils import (
    materialize_split,
    tooth_class_names,
    write_data_yaml,
)


SPLITS_DIR = Path(os.environ.get("PBL4_SPLITS_DIR", "data/splits"))
CLASS_MAP_PATH = Path(os.environ.get("PBL4_CLASS_MAP_PATH", "data/splits/class_map.txt"))
YOLO_SEG_ROOT = Path(os.environ.get("PBL4_YOLO_SEG_DIR", "data/yolo_seg"))
FOLDS = 4


def _infer_num_classes(class_map_path):
    max_id = 0
    for line in Path(class_map_path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            max_id = max(max_id, int(parts[-1]))
        except ValueError:
            continue
    if max_id == 0:
        raise SystemExit(f"Could not infer class count from {class_map_path}")
    return max_id + 1


def prepare_test_split(num_classes, min_area):
    test_root = YOLO_SEG_ROOT / "test"
    stats = materialize_split(
        SPLITS_DIR / "test" / "img",
        SPLITS_DIR / "test" / "masks_semantic",
        test_root / "images" / "test",
        test_root / "labels" / "test",
        num_classes,
        min_area=min_area,
    )
    print(f"  test: {stats}")
    return test_root, stats


def prepare_fold(fold_index, num_classes, min_area, test_rel):
    fold_src = SPLITS_DIR / "folds" / f"fold_{fold_index}"
    fold_dst = YOLO_SEG_ROOT / f"fold_{fold_index}"
    train_stats = materialize_split(
        fold_src / "train" / "img",
        fold_src / "train" / "masks_semantic",
        fold_dst / "images" / "train",
        fold_dst / "labels" / "train",
        num_classes,
        min_area=min_area,
    )
    val_stats = materialize_split(
        fold_src / "val" / "img",
        fold_src / "val" / "masks_semantic",
        fold_dst / "images" / "val",
        fold_dst / "labels" / "val",
        num_classes,
        min_area=min_area,
    )
    class_names = tooth_class_names(num_classes)
    yaml_path = write_data_yaml(
        fold_dst / "data.yaml",
        dataset_root=fold_dst,
        class_names=class_names,
        train="images/train",
        val="images/val",
        test=test_rel,
    )
    print(f"  fold {fold_index}: train={train_stats} val={val_stats}")
    return {
        "fold": fold_index,
        "data_yaml": str(yaml_path),
        "train": train_stats,
        "val": val_stats,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize YOLO-seg dataset views from data/splits."
    )
    parser.add_argument(
        "--folds", type=int, nargs="*", default=list(range(FOLDS)),
        help="Which folds to build (default: all 4).",
    )
    parser.add_argument(
        "--min-area", type=int, default=None,
        help="Min contour area when converting masks to polygons (default: preset).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preset = get_yolo_seg_preset()
    min_area = args.min_area if args.min_area is not None else preset["min_polygon_area"]
    num_classes = _infer_num_classes(CLASS_MAP_PATH)
    print(f"num_classes={num_classes} (background + {num_classes - 1} teeth)")

    YOLO_SEG_ROOT.mkdir(parents=True, exist_ok=True)
    print("Materializing fixed test split...")
    test_root, test_stats = prepare_test_split(num_classes, min_area)
    # data.yaml lives in each fold dir; reference the shared test split by abs path.
    test_rel = str((test_root / "images" / "test").resolve())

    fold_records = []
    for fold_index in args.folds:
        print(f"Materializing fold {fold_index}...")
        fold_records.append(prepare_fold(fold_index, num_classes, min_area, test_rel))

    write_json(
        YOLO_SEG_ROOT / "prepare_summary.json",
        {
            "num_classes": num_classes,
            "min_polygon_area": min_area,
            "test": {"root": str(test_root), **test_stats},
            "folds": fold_records,
        },
    )
    print(f"Done. Dataset views under {YOLO_SEG_ROOT}/")


if __name__ == "__main__":
    main()
