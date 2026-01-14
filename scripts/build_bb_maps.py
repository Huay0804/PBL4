import argparse
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = (".png",)


def infer_num_classes(class_map_path):
    path = Path(class_map_path)
    if not path.exists():
        return None
    max_id = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            max_id = max(max_id, int(parts[-1]))
        except ValueError:
            continue
    return max_id + 1 if max_id > 0 else None


def scan_num_classes(mask_dirs):
    max_id = 0
    for masks_dir in mask_dirs:
        for mask_path in masks_dir.glob("*.png"):
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            max_id = max(max_id, int(mask.max()))
    return max_id + 1 if max_id > 0 else None


def build_bb_map(mask, num_classes):
    height, width = mask.shape[:2]
    bb_map = np.zeros((height, width, num_classes), dtype=np.uint8)
    for class_id in range(1, num_classes):
        ys, xs = np.where(mask == class_id)
        if ys.size == 0:
            continue
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        bb_map[y1 : y2 + 1, x1 : x2 + 1, class_id] = 1
    return bb_map


def parse_args():
    parser = argparse.ArgumentParser(description="Build BB-Conv maps from semantic masks.")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--class-map", default="data/splits/class_map.txt")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    output_root = Path(args.output_dir) if args.output_dir else splits_dir

    mask_dirs = [splits_dir / split / "masks_semantic" for split in args.splits]
    num_classes = infer_num_classes(args.class_map)
    if num_classes is None:
        num_classes = scan_num_classes(mask_dirs)
    if num_classes is None:
        raise SystemExit("Unable to infer num_classes from class_map or masks.")

    total = 0
    for split in args.splits:
        masks_dir = splits_dir / split / "masks_semantic"
        if not masks_dir.exists():
            raise SystemExit(f"Missing masks directory: {masks_dir}")
        out_dir = output_root / split / "bb_maps"
        out_dir.mkdir(parents=True, exist_ok=True)

        for mask_path in masks_dir.glob("*.png"):
            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            bb_map = build_bb_map(mask, num_classes)
            out_path = out_dir / f"{mask_path.stem}.npz"
            np.savez_compressed(out_path, bb=bb_map)
            total += 1

    print(f"Wrote {total} BB maps with {num_classes} channels to {output_root}.")


if __name__ == "__main__":
    main()
