import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
MASKS_DIR_NAME = "masks_semantic"

IMAGES_DIR = Path("data/raw/img")
MASKS_DIR = Path("data/processed/masks_semantic")
CLASS_MAP = Path("data/processed/masks_semantic/class_map.txt")
OUTPUT_DIR = Path("data/splits")
TEST_COUNT = 111
TEST_RATIO = None
FOLDS = 4
SEED = 13
MODE = "copy"
INCLUDE_BACKGROUND = False
BALANCE_MODE = "pixels"
IMAGE_BALANCE_WEIGHT = 0.25
DRY_RUN = False


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Split dataset into test + k-fold CV.")
    parser.add_argument("--test-count", type=int, default=TEST_COUNT)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    parser.add_argument("--folds", type=int, default=FOLDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mode", type=str, default=MODE, choices=("copy", "symlink"))
    parser.add_argument("--include-background", action="store_true", default=INCLUDE_BACKGROUND)
    parser.add_argument("--balance-mode", type=str, default=BALANCE_MODE, choices=("pixels", "presence"))
    parser.add_argument("--image-balance-weight", type=float, default=IMAGE_BALANCE_WEIGHT)
    parser.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    return parser.parse_args()


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


def sum_counts(items, num_classes):
    total = np.zeros(num_classes, dtype=np.int64)
    for item in items:
        total += item["counts"]
    return total


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


def select_balanced_subset(samples, count):
    if count <= 0:
        return [], list(samples)
    if count >= len(samples):
        return list(samples), []

    total_counts = sum_counts(samples, samples[0]["counts"].shape[0])
    total_sum = float(total_counts.sum())
    if total_sum > 0:
        mask = total_counts > 0
        target = total_counts[mask] / total_sum
    else:
        mask = None
        target = None

    selected = []
    remaining = list(samples)
    selected_counts = np.zeros_like(total_counts)

    for _ in range(count):
        best_idx = None
        best_score = None
        for idx, sample in enumerate(remaining):
            new_counts = selected_counts + sample["counts"]
            new_total = float(new_counts.sum()) + 1e-6
            if total_sum > 0:
                dist = new_counts[mask] / new_total
                score = float(np.sum(np.abs(dist - target) * target))
            else:
                score = 0.0

            if best_score is None or score < best_score - 1e-12:
                best_score = score
                best_idx = idx
            elif abs(score - best_score) <= 1e-12:
                if sample["image"].name < remaining[best_idx]["image"].name:
                    best_idx = idx

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        selected_counts += chosen["counts"]

    return selected, remaining


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


def summarize(split_counts, ratios, balance_mode, label_prefix=""):
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
        prefix = f"{label_prefix} " if label_prefix else ""
        print(
            f"{prefix}{name}: {label}={int(counts.sum())} "
            f"dist_l1={score:.4f} "
            f"classes={dist.round(4).tolist()}"
        )


def main():
    args = parse_args()

    if args.folds < 2:
        raise ValueError("folds must be >= 2.")

    pairs = list_pairs(IMAGES_DIR, MASKS_DIR)
    if not pairs:
        raise SystemExit("No image/mask pairs found.")

    num_classes = infer_num_classes(CLASS_MAP, MASKS_DIR)

    samples = []
    for img_path, mask_path in pairs:
        counts = mask_counts(
            mask_path,
            num_classes,
            include_background=args.include_background,
            balance_mode=args.balance_mode,
        )
        samples.append({"image": img_path, "mask": mask_path, "counts": counts})

    # Shuffle first (for deterministic tie-breaking), then stable-sort by class richness.
    random.Random(args.seed).shuffle(samples)
    samples.sort(
        key=lambda s: (-int(s["counts"].sum()), -int(np.count_nonzero(s["counts"])))
    )

    total_samples = len(samples)
    if args.test_count is not None:
        if args.test_count <= 0 or args.test_count >= total_samples:
            raise ValueError("test-count must be between 1 and total_samples - 1.")
        test_samples, train_samples = select_balanced_subset(samples, args.test_count)
        split_counts = {
            "test": sum_counts(test_samples, num_classes),
            "train_pool": sum_counts(train_samples, num_classes),
        }
        split_ratios = {
            "test": args.test_count / total_samples,
            "train_pool": 1.0 - (args.test_count / total_samples),
        }
    else:
        test_ratio = args.test_ratio if args.test_ratio is not None else 0.2
        if test_ratio <= 0 or test_ratio >= 1:
            raise ValueError("test-ratio must be between 0 and 1.")
        split_ratios = {"test": test_ratio, "train_pool": 1.0 - test_ratio}
        split_samples, split_counts = assign_splits(samples, split_ratios, args.image_balance_weight)
        test_samples = split_samples["test"]
        train_samples = split_samples["train_pool"]

    fold_ratios = {f"fold_{i}": 1.0 / args.folds for i in range(args.folds)}
    fold_samples, fold_counts = assign_splits(train_samples, fold_ratios, args.image_balance_weight)

    output_dir = OUTPUT_DIR
    if output_dir.exists() and not args.dry_run:
        resolved = output_dir.resolve()
        if resolved in (Path("/"), Path.home(), Path.cwd()):
            raise ValueError(f"Refusing to clear unsafe output dir: {resolved}")
        shutil.rmtree(output_dir)
    if not args.dry_run:
        (output_dir / "folds").mkdir(parents=True, exist_ok=True)

    def _write_samples(items, root_dir):
        img_dir = root_dir / "img"
        mask_dir = root_dir / MASKS_DIR_NAME
        if not args.dry_run:
            img_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
        for sample in items:
            if args.dry_run:
                continue
            copy_item(sample["image"], img_dir / sample["image"].name, args.mode)
            copy_item(sample["mask"], mask_dir / sample["mask"].name, args.mode)

    _write_samples(test_samples, output_dir / "test")

    fold_names = sorted(fold_samples.keys())
    for fold_name in fold_names:
        val_items = fold_samples[fold_name]
        train_items = [
            s for other_name in fold_names if other_name != fold_name
            for s in fold_samples[other_name]
        ]
        fold_root = output_dir / "folds" / fold_name
        _write_samples(train_items, fold_root / "train")
        _write_samples(val_items, fold_root / "val")

    if CLASS_MAP.exists() and not args.dry_run:
        shutil.copy2(CLASS_MAP, output_dir / "class_map.txt")

    print(f"Split summary (balance: {args.balance_mode}, folds={args.folds}):")
    print(f"test: {len(test_samples)} samples")
    print(f"train_pool: {len(train_samples)} samples")
    for fold_name in fold_names:
        print(f"{fold_name} val: {len(fold_samples[fold_name])} samples")
    summarize(split_counts, split_ratios, args.balance_mode, label_prefix="test/train_pool")
    summarize(fold_counts, fold_ratios, args.balance_mode, label_prefix="folds")


if __name__ == "__main__":
    main()
