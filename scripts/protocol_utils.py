import json
from pathlib import Path

import numpy as np
from PIL import Image


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _dedupe_paths(paths):
    seen = set()
    ordered = []
    for path in paths:
        # strict=False: normalize without requiring the path to already exist.
        key = str(Path(path).resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(Path(path))
    return ordered


def _iter_segmentation_run_dirs(run_dir):
    run_dir = Path(run_dir)
    candidates = []

    latest_dir = run_dir / "latest"
    if latest_dir.exists() and latest_dir.is_dir():
        candidates.append(latest_dir.resolve())

    latest_meta_path = run_dir / "latest_run.json"
    latest_meta = load_json(latest_meta_path, default={}) or {}
    meta_path = latest_meta.get("latest_run_dir") or latest_meta.get("run_dir")
    if meta_path:
        resolved = Path(meta_path)
        if not resolved.is_absolute():
            resolved = (run_dir / resolved).resolve()
        if resolved.exists() and resolved.is_dir():
            candidates.append(resolved)

    if run_dir.exists():
        subdirs = []
        for child in run_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in {"latest", "logs", "__pycache__"}:
                continue
            if (child / "best.keras").exists() or (child / "last.keras").exists():
                subdirs.append(child)
        subdirs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        candidates.extend(subdirs)

    candidates.append(run_dir)
    return _dedupe_paths(candidates)


def resolve_segmentation_checkpoint(run_dir, checkpoint_policy="best", checkpoint_path=None):
    run_dir = Path(run_dir)
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path)
        if not resolved.exists():
            raise SystemExit(f"Checkpoint not found: {resolved}")
        return resolved, {"requested_policy": "explicit", "selection": "explicit_path"}

    if checkpoint_policy not in {"best", "last"}:
        raise ValueError(f"Unknown checkpoint policy: {checkpoint_policy}")

    searched = []
    for candidate_run_dir in _iter_segmentation_run_dirs(run_dir):
        if checkpoint_policy == "best":
            candidates = [
                (candidate_run_dir / "best.keras", "best"),
                (candidate_run_dir / "last.keras", "best_fallback_last"),
            ]
        else:
            candidates = [(candidate_run_dir / "last.keras", "last")]

        for path, selection in candidates:
            searched.append(str(path))
            if path.exists():
                return path, {
                    "requested_policy": checkpoint_policy,
                    "selection": selection,
                    "checkpoint_root": str(candidate_run_dir),
                }

    searched_str = "\n  - ".join(searched)
    raise SystemExit(
        f"No segmentation checkpoint found in {run_dir} for policy '{checkpoint_policy}'. "
        f"Searched:\n  - {searched_str}"
    )


def merge_fold_rows(existing_rows, new_rows):
    merged = {}
    for row in existing_rows or []:
        merged[int(row["fold"])] = row
    for row in new_rows or []:
        merged[int(row["fold"])] = row
    return [merged[idx] for idx in sorted(merged.keys())]


def pair_stems(pairs):
    stems = set()
    for pair in pairs:
        stems.add(Path(pair[0]).stem)
    return stems


def validate_disjoint_pair_sets(named_pairs):
    names = list(named_pairs.keys())
    for i, left_name in enumerate(names):
        left = pair_stems(named_pairs[left_name])
        for right_name in names[i + 1 :]:
            right = pair_stems(named_pairs[right_name])
            overlap = sorted(left & right)
            if overlap:
                sample = ", ".join(overlap[:5])
                raise SystemExit(
                    f"Split integrity error: {left_name} and {right_name} overlap "
                    f"on {len(overlap)} stems (e.g. {sample})."
                )


def summarize_mask_paths(mask_paths, num_classes, subset_name):
    empty_stems = []
    max_label = 0
    for mask_path in mask_paths:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.size == 0:
            empty_stems.append(Path(mask_path).stem)
            continue
        current_max = int(mask.max())
        if current_max >= num_classes:
            raise SystemExit(
                f"Class map mismatch in {subset_name}: mask {mask_path} has label "
                f"{current_max}, but expected labels < {num_classes}."
            )
        max_label = max(max_label, current_max)
        if current_max == 0:
            empty_stems.append(Path(mask_path).stem)

    return {
        "count": len(mask_paths),
        "max_label": max_label,
        "empty_mask_count": len(empty_stems),
        "empty_mask_stems": empty_stems,
    }


def validate_bb_map_files(bb_paths, expected_height, expected_width, expected_channels, subset_name):
    nonzero_files = 0
    expected_shape = (expected_height, expected_width, expected_channels)
    for bb_path in bb_paths:
        data = np.load(bb_path)
        if isinstance(data, np.lib.npyio.NpzFile):
            if "bb" in data:
                array = data["bb"]
            else:
                array = data[data.files[0]]
        else:
            array = data
        if tuple(array.shape) != expected_shape:
            raise SystemExit(
                f"BB-map shape mismatch in {subset_name}: {bb_path} has shape "
                f"{tuple(array.shape)}, expected {expected_shape}."
            )
        nonzero_files += int(np.any(array))

    return {
        "count": len(bb_paths),
        "expected_shape": list(expected_shape),
        "nonzero_files": nonzero_files,
    }
