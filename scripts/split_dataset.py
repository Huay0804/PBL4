import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MASKS_DIR_NAME = "masks_semantic"


class _SingleUseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) is not None:
            parser.error(f"{option_string} specified multiple times")
        setattr(namespace, self.dest, values)


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test folders.")
    parser.add_argument("--images-dir", default="data/raw/img")
    parser.add_argument("--masks-dir", default="data/processed/masks_semantic")
    parser.add_argument("--class-map", default="data/processed/masks_semantic/class_map.txt")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--include-background", action="store_true")
    parser.add_argument(
        "--balance-mode",
        choices=["pixels", "presence"],
        action=_SingleUseAction,
        default=None,
    )
    parser.add_argument("--image-balance-weight", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.balance_mode is None:
        args.balance_mode = "pixels"
    return args


def infer_num_classes(class_map_path, masks_dir):
    path = Path(class_map_path)
    if path.exists():
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
        return max_id + 1

    max_id = 0
    for mask_path in Path(masks_dir).glob("*.png"):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        max_id = max(max_id, int(mask.max()))
    return max_id + 1


def list_pairs(images_dir, masks_dir):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    masks = {p.stem: p for p in masks_dir.glob("*.png")}
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    pairs = []
    missing = 0
    for img in sorted(image_paths):
        mask = masks.get(img.stem)
        if mask is None:
            missing += 1
            continue
        pairs.append((img, mask))
    if missing:
        print(f"warning: {missing} images without masks were skipped", file=sys.stderr)
    return pairs


def mask_counts(mask_path, num_classes, include_background, balance_mode):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    counts = np.bincount(mask.reshape(-1), minlength=num_classes).astype(np.int64)
    if not include_background and num_classes > 1:
        counts[0] = 0
    if balance_mode == "presence":
        counts = (counts > 0).astype(np.int64)
    return counts


def validate_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("Sum of ratios must be > 0.")
    if abs(total - 1.0) > 1e-6:
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
    return train_ratio, val_ratio, test_ratio


def assign_splits(samples, ratios, image_weight):
    split_names = list(ratios.keys())
    num_splits = len(split_names)
    total_images = len(samples)

    total_counts = np.zeros_like(samples[0]["counts"])
    for sample in samples:
        total_counts += sample["counts"]
    total_counts_sum = float(total_counts.sum())

    target_images = {
        name: ratios[name] * total_images for name in split_names
    }
    if total_counts_sum > 0:
        mask = total_counts > 0
        weights = total_counts[mask] / total_counts_sum
    else:
        mask = None
        weights = None

    split_counts = {name: np.zeros_like(total_counts) for name in split_names}
    split_images = {name: 0 for name in split_names}
    split_samples = {name: [] for name in split_names}

    for sample in samples:
        best_name = None
        best_score = None
        sample_counts = sample["counts"]
        sample_sum = float(sample_counts.sum()) + 1e-6
        for name in split_names:
            score = 0.0
            if total_counts_sum > 0:
                current_share = split_counts[name][mask] / total_counts[mask]
                deficit = ratios[name] - current_share
                deficit = np.clip(deficit, 0.0, None)
                score += float(np.sum(deficit * sample_counts[mask]) / sample_sum)

            image_deficit = ratios[name] - (split_images[name] / total_images)
            if image_deficit > 0:
                score += image_weight * image_deficit

            if best_score is None or score > best_score:
                best_score = score
                best_name = name
        split_counts[best_name] += sample_counts
        split_images[best_name] += 1
        split_samples[best_name].append(sample)

    return split_samples, split_counts


def copy_item(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unknown mode {mode}")


def summarize(split_counts, ratios, balance_mode):
    total_counts = np.zeros_like(next(iter(split_counts.values())))
    for counts in split_counts.values():
        total_counts += counts
    total_sum = total_counts.sum()
    if total_sum == 0:
        print("warning: no foreground pixels found; balance check skipped")
        return
    mask = total_counts > 0
    weights = total_counts[mask] / (total_sum + 1e-6)
    for name, counts in split_counts.items():
        counts_sel = counts[mask]
        dist = counts_sel / (counts_sel.sum() + 1e-6)
        share = counts_sel / total_counts[mask]
        diff = np.abs(share - ratios[name])
        score = float(np.sum(diff * weights))
        label = "pixels" if balance_mode == "pixels" else "presence"
        print(
            f"{name}: {label}={int(counts.sum())} "
            f"dist_l1={score:.4f} "
            f"classes={dist.round(4).tolist()}"
        )


def main():
    args = parse_args()
    train_ratio, val_ratio, test_ratio = validate_ratios(
        args.train_ratio, args.val_ratio, args.test_ratio
    )
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    pairs = list_pairs(args.images_dir, args.masks_dir)
    if not pairs:
        raise SystemExit("No image/mask pairs found.")

    num_classes = infer_num_classes(args.class_map, args.masks_dir)

    samples = []
    for img_path, mask_path in pairs:
        counts = mask_counts(
            mask_path,
            num_classes,
            include_background=args.include_background,
            balance_mode=args.balance_mode,
        )
        samples.append(
            {
                "image": img_path,
                "mask": mask_path,
                "counts": counts,
            }
        )

    samples.sort(
        key=lambda s: (-int(s["counts"].sum()), -int(np.count_nonzero(s["counts"])), s["image"].name)
    )
    random.Random(args.seed).shuffle(samples)
    samples.sort(
        key=lambda s: (-int(s["counts"].sum()), -int(np.count_nonzero(s["counts"])), s["image"].name)
    )

    split_samples, split_counts = assign_splits(samples, ratios, args.image_balance_weight)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.dry_run:
        resolved = output_dir.resolve()
        if resolved in (Path("/"), Path.home(), Path.cwd()):
            raise ValueError(f"Refusing to clear unsafe output dir: {resolved}")
        shutil.rmtree(output_dir)
    for split_name, items in split_samples.items():
        img_dir = output_dir / split_name / "img"
        mask_dir = output_dir / split_name / MASKS_DIR_NAME
        if not args.dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
        for sample in items:
            if args.dry_run:
                continue
            copy_item(sample["image"], img_dir / sample["image"].name, args.mode)
            copy_item(sample["mask"], mask_dir / sample["mask"].name, args.mode)

    class_map_path = Path(args.class_map)
    if class_map_path.exists() and not args.dry_run:
        shutil.copy2(class_map_path, output_dir / "class_map.txt")

    print(f"Split summary (balance: {args.balance_mode}):")
    for name, items in split_samples.items():
        print(f"{name}: {len(items)} samples")
    summarize(split_counts, ratios, args.balance_mode)


if __name__ == "__main__":
    main()
