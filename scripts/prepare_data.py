import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

SRC_ROOT = Path("data/raw/archive/Teeth Segmentation PNG/d2")
DST_ROOT = Path("data")
USE_MACHINE_MASKS = False
RAW_DIR = "raw"
PROCESSED_DIR = "processed"
DIRS_TO_COPY = [
    "img",
    "ann",
    "masks_human",
]


def copy_dir(src_root: Path, dst_root: Path, name: str) -> bool:
    src_dir = src_root / name
    if not src_dir.exists():
        print(f"missing dir: {src_dir}", file=sys.stderr)
        return False
    dst_dir = dst_root / name
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
    print(f"copied {src_dir} -> {dst_dir}")
    return True


def sort_key(value: str) -> tuple[int, str]:
    try:
        return (0, int(value))
    except Exception:
        return (1, str(value))


def make_palette(size: int = 256) -> list[int]:
    palette = [0] * (size * 3)
    for i in range(size):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        palette[i * 3 + 0] = r
        palette[i * 3 + 1] = g
        palette[i * 3 + 2] = b
    return palette


def write_vis_from_masks(mask_dir: Path, vis_dir: Path) -> None:
    palette = make_palette()
    vis_dir.mkdir(parents=True, exist_ok=True)
    for mask_path in sorted(mask_dir.glob("*.png")):
        with Image.open(mask_path) as mask:
            vis = mask.convert("P")
            vis.putpalette(palette)
            vis.save(vis_dir / mask_path.name)


def generate_masks(ann_dir: Path, out_dir: Path, vis_dir: Path) -> int:
    if not ann_dir.exists():
        print(f"annotations not found: {ann_dir}", file=sys.stderr)
        return 1

    ann_files = sorted(ann_dir.glob("*.json"))
    if not ann_files:
        print(f"no annotation files in: {ann_dir}", file=sys.stderr)
        return 1

    classes: set[str] = set()
    for ann in ann_files:
        data = json.loads(ann.read_text())
        for obj in data.get("objects", []):
            cls = obj.get("classTitle")
            if cls is not None:
                classes.add(cls)

    if not classes:
        print(f"no classes found in: {ann_dir}", file=sys.stderr)
        return 1

    sorted_classes = sorted(classes, key=sort_key)
    class_map = {cls: i + 1 for i, cls in enumerate(sorted_classes)}

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "class_map.txt").write_text(
        "\n".join(f"{k}\t{v}" for k, v in class_map.items()) + "\n"
    )

    for ann in ann_files:
        name = ann.name
        stem = name[:-9] if name.endswith(".jpg.json") else ann.stem
        data = json.loads(ann.read_text())
        size = data["size"]
        w, h = int(size["width"]), int(size["height"])
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        for obj in data.get("objects", []):
            cls = obj.get("classTitle")
            if cls not in class_map:
                continue
            pts = obj.get("points", {}).get("exterior", [])
            if len(pts) < 3:
                continue
            poly = [(float(x), float(y)) for x, y in pts]
            draw.polygon(poly, fill=class_map[cls])

            for hole in obj.get("points", {}).get("interior", []) or []:
                if len(hole) >= 3:
                    hpoly = [(float(x), float(y)) for x, y in hole]
                    draw.polygon(hpoly, fill=0)

        mask.save(out_dir / f"{stem}.png")

    write_vis_from_masks(out_dir, vis_dir)

    return 0


def derive_color_to_id_map(rgb_dir: Path, id_dir: Path) -> tuple[dict[int, int], dict[int, dict[int, int]]]:
    try:
        import numpy as np
    except Exception as exc:
        print(f"numpy is required for mask conversion: {exc}", file=sys.stderr)
        return {}, {}

    counts: dict[int, Counter] = {}
    missing_pairs = 0

    for rgb_path in sorted(rgb_dir.glob("*.png")):
        id_path = id_dir / rgb_path.name
        if not id_path.exists():
            missing_pairs += 1
            continue

        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        ids = np.array(Image.open(id_path).convert("L"))
        if rgb.shape[:2] != ids.shape:
            print(f"size mismatch: {rgb_path} vs {id_path}", file=sys.stderr)
            continue

        rgb_int = (
            (rgb[:, :, 0].astype("uint32") << 16)
            | (rgb[:, :, 1].astype("uint32") << 8)
            | rgb[:, :, 2].astype("uint32")
        )
        pair = (rgb_int.astype("uint64") << 8) | ids.astype("uint64")
        uniq, cnt = np.unique(pair.reshape(-1), return_counts=True)
        for u, c in zip(uniq, cnt):
            color = int(u >> 8)
            label = int(u & 0xFF)
            if color not in counts:
                counts[color] = Counter()
            counts[color][label] += int(c)

    if missing_pairs:
        print(f"missing id masks for {missing_pairs} RGB masks", file=sys.stderr)

    mapping: dict[int, int] = {}
    ambiguous: dict[int, dict[int, int]] = {}
    for color, counter in counts.items():
        label, best = counter.most_common(1)[0]
        total = sum(counter.values())
        if len(counter) > 1 and (best / total) < 0.98:
            ambiguous[color] = dict(counter)
        mapping[color] = label

    if 0 not in mapping:
        mapping[0] = 0

    return mapping, ambiguous


def write_color_to_id_map(path: Path, mapping: dict[int, int]) -> None:
    data = {}
    for color, label in sorted(mapping.items()):
        hex_color = f"#{color:06X}"
        data[hex_color] = label
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def convert_rgb_masks(rgb_dir: Path, out_dir: Path, mapping: dict[int, int]) -> Counter:
    try:
        import numpy as np
    except Exception as exc:
        print(f"numpy is required for mask conversion: {exc}", file=sys.stderr)
        return Counter()

    unknown = Counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    for rgb_path in sorted(rgb_dir.glob("*.png")):
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        rgb_int = (
            (rgb[:, :, 0].astype("uint32") << 16)
            | (rgb[:, :, 1].astype("uint32") << 8)
            | rgb[:, :, 2].astype("uint32")
        )
        out = np.zeros(rgb_int.shape, dtype="uint8")
        for color in np.unique(rgb_int):
            label = mapping.get(int(color))
            if label is None:
                unknown[int(color)] += int((rgb_int == color).sum())
                label = 0
            out[rgb_int == color] = label
        Image.fromarray(out, mode="L").save(out_dir / rgb_path.name)
    return unknown


def main() -> int:
    src_root = SRC_ROOT
    dst_root = DST_ROOT
    raw_root = dst_root / RAW_DIR
    processed_root = dst_root / PROCESSED_DIR

    if not src_root.exists():
        print(f"source not found: {src_root}", file=sys.stderr)
        return 1

    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in DIRS_TO_COPY:
        if not copy_dir(src_root, raw_root, name):
            missing.append(name)
    if missing:
        print(f"missing required directories: {', '.join(missing)}", file=sys.stderr)
        return 1

    if USE_MACHINE_MASKS:
        if not copy_dir(src_root, raw_root, "masks_machine"):
            print("missing required directory: masks_machine", file=sys.stderr)
            return 1

    ann_dir = raw_root / "ann"
    out_dir = processed_root / "masks_semantic"
    vis_dir = processed_root / "masks_semantic_vis"
    result = generate_masks(ann_dir, out_dir, vis_dir)
    if result != 0:
        return result

    if USE_MACHINE_MASKS:
        rgb_dir = raw_root / "masks_machine"
        mapping, ambiguous = derive_color_to_id_map(rgb_dir, out_dir)
        if not mapping:
            print("failed to derive color mapping for masks_machine", file=sys.stderr)
            return 1
        write_color_to_id_map(out_dir / "machine_color_to_id.json", mapping)
        unknown = convert_rgb_masks(rgb_dir, out_dir, mapping)
        write_vis_from_masks(out_dir, vis_dir)
        if ambiguous:
            sample = list(ambiguous.items())[:5]
            print(f"warning: ambiguous color mappings (sample): {sample}", file=sys.stderr)
        if unknown:
            sample = list(unknown.items())[:5]
            print(f"warning: unmapped colors (sample): {sample}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
