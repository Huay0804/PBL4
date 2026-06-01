"""Shared helpers for the YOLO instance-segmentation baselines.

The dense segmenters (TransUNet, mod_nestnet, icpr_munet) are trained and scored
on per-pixel semantic masks. Ultralytics YOLO -seg models instead predict one
mask per detected tooth. To run them under the *same* protocol we:

  1. convert each semantic mask PNG into YOLO polygon labels (one line per tooth
     instance: ``cls x1 y1 x2 y2 ...`` normalized to [0, 1]), and
  2. at evaluation time rasterize the predicted instance masks back into a single
     33-class label map, so the existing confusion-matrix / per-class / per-group
     IoU+Dice reporting in ``train.py`` applies unchanged.

Because each FDI tooth class appears at most once per panoramic image, the
conversion is near loss-less: one connected contour per class value.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_image_mask_pairs(images_dir, masks_dir):
    """Match images to their semantic mask by filename stem (sorted)."""
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    masks = {p.stem: p for p in masks_dir.glob("*.png")}
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    pairs = []
    for img in sorted(image_paths):
        mask = masks.get(img.stem)
        if mask is not None:
            pairs.append((img, mask))
    return pairs


def _read_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def mask_to_polygons(mask, num_classes, min_area=16):
    """Convert a single-channel label mask into YOLO-seg polygon rows.

    Returns a list of ``(class_id_0based, [x1, y1, x2, y2, ...])`` where the
    coordinates are normalized to [0, 1] by the mask width/height. Background
    (label 0) is skipped. One row is emitted per connected component so a tooth
    that is split by occlusion still yields valid (multi-instance) labels.
    """
    height, width = mask.shape[:2]
    rows = []
    present = [int(v) for v in np.unique(mask) if int(v) != 0 and int(v) < num_classes]
    for value in present:
        binary = (mask == value).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            pts = contour.reshape(-1, 2).astype(np.float64)
            if pts.shape[0] < 3:
                continue
            pts[:, 0] /= float(width)
            pts[:, 1] /= float(height)
            np.clip(pts, 0.0, 1.0, out=pts)
            rows.append((value - 1, pts.reshape(-1).tolist()))
    return rows


def write_yolo_label(label_path, rows):
    """Write polygon rows to a YOLO ``.txt`` label (empty file if no teeth)."""
    label_path = Path(label_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for class_id, coords in rows:
        coord_str = " ".join(f"{c:.6f}" for c in coords)
        lines.append(f"{class_id} {coord_str}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def convert_mask_file(mask_path, label_path, num_classes, min_area=16):
    mask = _read_mask(mask_path)
    rows = mask_to_polygons(mask, num_classes, min_area=min_area)
    write_yolo_label(label_path, rows)
    return len(rows)


def _link_or_copy(src, dst):
    """Symlink an image into the dataset view; copy as a fallback."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(Path(src).resolve())
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def materialize_split(images_dir, masks_dir, dst_images, dst_labels,
                      num_classes, min_area=16):
    """Build one YOLO split: link images, write polygon labels. Returns stats."""
    dst_images = Path(dst_images)
    dst_labels = Path(dst_labels)
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)
    pairs = list_image_mask_pairs(images_dir, masks_dir)
    instance_count = 0
    empty = 0
    for img, mask in pairs:
        _link_or_copy(img, dst_images / img.name)
        n = convert_mask_file(mask, dst_labels / f"{img.stem}.txt", num_classes,
                              min_area=min_area)
        instance_count += n
        if n == 0:
            empty += 1
    return {"images": len(pairs), "instances": instance_count, "empty_labels": empty}


def write_data_yaml(yaml_path, dataset_root, class_names, train="images/train",
                    val="images/val", test=None):
    """Write an Ultralytics ``data.yaml`` for a fold-local dataset view."""
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {Path(dataset_root).resolve()}",
        f"train: {train}",
        f"val: {val}",
    ]
    if test is not None:
        lines.append(f"test: {test}")
    lines.append(f"nc: {len(class_names)}")
    names_str = ", ".join(f"{i}: {name}" for i, name in enumerate(class_names))
    lines.append(f"names: {{{names_str}}}")
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


def tooth_class_names(num_classes):
    """Ordered 0-based YOLO class names: tooth_1 .. tooth_32 (no background)."""
    return [f"tooth_{i}" for i in range(1, num_classes)]


def rasterize_result(result, height, width, num_classes, conf_threshold=0.25):
    """Collapse an Ultralytics seg ``Results`` into a (H, W) label map.

    Masks are painted in ascending-confidence order so the highest-confidence
    tooth wins any pixel overlap. Predicted class ``c`` (0-based) maps to label
    ``c + 1``; background stays 0. Returns int32 array of shape (H, W).
    """
    label_map = np.zeros((height, width), dtype=np.int32)
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or boxes is None or masks.data is None or len(boxes) == 0:
        return label_map

    mask_data = masks.data.cpu().numpy()           # (N, mh, mw) in [0, 1]
    classes = boxes.cls.cpu().numpy().astype(int)  # (N,)
    confs = boxes.conf.cpu().numpy()               # (N,)

    order = np.argsort(confs)  # low -> high, so high conf painted last
    for idx in order:
        if confs[idx] < conf_threshold:
            continue
        m = mask_data[idx]
        if m.shape != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
        binary = m >= 0.5
        if not binary.any():
            continue
        label_map[binary] = classes[idx] + 1
    return label_map


def load_semantic_mask(mask_path, height, width):
    """Load a GT semantic mask resized (nearest) to the eval resolution.

    Matches make_dataset's nearest-neighbour resize so YOLO and the dense
    segmenters are scored on the same (height, width) grid.
    """
    mask = _read_mask(mask_path)
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.int32)
