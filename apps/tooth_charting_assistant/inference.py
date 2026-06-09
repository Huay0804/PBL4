"""Inference backends for the Tooth Charting Assistant.

Two backends produce the same `Prediction` shape:
  * YoloSegBackend  — Ultralytics YOLO-seg (.pt). Instance masks are collapsed
    into one 33-class semantic label map using the SAME rule as
    scripts/yolo_seg_utils.rasterize_result: paint masks in ascending-confidence
    order so the most confident tooth wins overlaps, predicted class c -> label
    c + 1, background stays 0.
  * DenseKerasBackend — optional. Loads an image-only .keras semantic segmenter
    (e.g. TransUNet) via TensorFlow if available; argmax over the softmax gives
    the label map. BB-gated models (>1 input) are reported as unsupported here.

Heavy deps (ultralytics, tensorflow) are imported lazily inside `load()` so the
app starts even when they are absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import NUM_CLASSES, INPUT_HEIGHT, INPUT_WIDTH, YOLO_IMGSZ, REPO_ROOT


@dataclass
class ToothInstance:
    tooth_id: int                      # 1..32 (semantic class = tooth_id)
    class_index: int                   # 0..31 (model/0-based)
    mask_area_pixels: int
    confidence: float | None = None
    bbox_xyxy: list[float] | None = None


@dataclass
class Prediction:
    label_map: np.ndarray              # (H, W) int32 in [0, NUM_CLASSES-1]
    instances: list[ToothInstance]
    height: int
    width: int
    model_name: str
    backend: str
    params: dict = field(default_factory=dict)

    @property
    def num_teeth(self) -> int:
        return len(self.instances)


class BackendError(RuntimeError):
    """Raised for actionable, user-facing inference failures."""


def _pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def _instances_from_label_map(
    label_map: np.ndarray,
    conf_by_class: dict[int, float] | None = None,
    bbox_by_class: dict[int, list[float]] | None = None,
) -> list[ToothInstance]:
    """One row per tooth class present (charting view), area from the final map."""
    instances = []
    present = [int(v) for v in np.unique(label_map) if int(v) != 0]
    for tooth_id in sorted(present):
        mask = label_map == tooth_id
        area = int(mask.sum())
        if area == 0:
            continue
        bbox = None
        if bbox_by_class and tooth_id in bbox_by_class:
            bbox = [round(float(x), 1) for x in bbox_by_class[tooth_id]]
        else:
            ys, xs = np.where(mask)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        conf = None
        if conf_by_class and tooth_id in conf_by_class:
            conf = round(float(conf_by_class[tooth_id]), 4)
        instances.append(
            ToothInstance(
                tooth_id=tooth_id,
                class_index=tooth_id - 1,
                mask_area_pixels=area,
                confidence=conf,
                bbox_xyxy=bbox,
            )
        )
    return instances


class YoloSegBackend:
    backend = "yolo"

    def __init__(self, checkpoint_path, model_name="YOLO-seg"):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self._model = None

    def load(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - env dependent
            raise BackendError(
                "Ultralytics is not installed. Run `pip install -r "
                "apps/tooth_charting_assistant/requirements.txt`."
            ) from exc
        if not self.checkpoint_path:
            raise BackendError("No checkpoint provided for the YOLO backend.")
        if not Path(self.checkpoint_path).exists():
            raise BackendError(f"Checkpoint not found: {self.checkpoint_path}")
        try:
            self._model = YOLO(str(self.checkpoint_path))
        except Exception as exc:  # pragma: no cover - env/file dependent
            raise BackendError(
                f"Failed to load YOLO checkpoint '{self.checkpoint_path}': {exc}. "
                "The file may be corrupt or built with an incompatible Ultralytics version."
            ) from exc

    def predict(self, image: Image.Image, conf: float, iou: float) -> Prediction:
        self.load()
        rgb = _pil_to_rgb_array(image)
        h0, w0 = rgb.shape[:2]
        result = self._model.predict(
            source=image.convert("RGB"),
            imgsz=YOLO_IMGSZ,
            conf=conf,
            iou=iou,
            retina_masks=True,
            verbose=False,
        )[0]

        label_map = np.zeros((h0, w0), dtype=np.int32)
        conf_by_class: dict[int, float] = {}
        bbox_by_class: dict[int, list[float]] = {}

        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is not None and boxes is not None and masks.data is not None and len(boxes):
            mask_data = masks.data.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            # Paint low -> high confidence so the most confident tooth wins overlaps.
            for idx in np.argsort(confs):
                if confs[idx] < conf:
                    continue
                m = mask_data[idx]
                if m.shape != (h0, w0):
                    m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
                binary = m >= 0.5
                if not binary.any():
                    continue
                tooth_id = int(classes[idx]) + 1
                # Guard against custom checkpoints whose class count differs from
                # the 32-tooth convention; DenseKerasBackend clamps similarly.
                if tooth_id <= 0 or tooth_id >= NUM_CLASSES:
                    continue
                label_map[binary] = tooth_id
                # Track the most confident detection per class for the table.
                if confs[idx] >= conf_by_class.get(tooth_id, -1.0):
                    conf_by_class[tooth_id] = float(confs[idx])
                    bbox_by_class[tooth_id] = [float(v) for v in xyxy[idx]]

        instances = _instances_from_label_map(label_map, conf_by_class, bbox_by_class)
        return Prediction(
            label_map=label_map,
            instances=instances,
            height=h0,
            width=w0,
            model_name=self.model_name,
            backend=self.backend,
            params={"conf": conf, "iou": iou, "imgsz": YOLO_IMGSZ},
        )


class DenseKerasBackend:
    """Optional image-only dense semantic segmenter (e.g. TransUNet)."""

    backend = "dense"

    def __init__(self, checkpoint_path, model_name="dense"):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self._model = None

    def load(self):
        if self._model is not None:
            return
        if not self.checkpoint_path:
            raise BackendError("No .keras checkpoint provided for the dense backend.")
        import sys

        # Make the repo's custom layers importable so load_model can deserialize.
        for path in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
            if path not in sys.path:
                sys.path.insert(0, path)
        try:
            import keras  # noqa: F401  (TF/Keras present?)
            try:
                import segmentation_models  # noqa: F401  registers custom classes
            except Exception:
                pass
            import keras as _keras
        except Exception as exc:  # pragma: no cover - env dependent
            raise BackendError(
                "TensorFlow/Keras is not available, so dense models can't run. "
                "Install the research environment, or use a YOLO model instead."
            ) from exc
        try:
            self._model = _keras.models.load_model(
                str(self.checkpoint_path), compile=False, safe_mode=False
            )
        except Exception as exc:
            raise BackendError(
                f"Failed to load dense checkpoint: {exc}. The MVP supports only "
                "image-only dense models (e.g. TransUNet)."
            ) from exc
        if len(getattr(self._model, "inputs", [1])) > 1:
            self._model = None
            raise BackendError(
                "This dense model expects bounding-box prior maps (it is BB-gated). "
                "Generating priors needs the YOLOX/Mask R-CNN pipeline, which is "
                "out of scope for this MVP. Use YOLO11-seg or TransUNet instead."
            )

    def predict(self, image: Image.Image, conf: float, iou: float) -> Prediction:
        self.load()
        rgb = _pil_to_rgb_array(image)
        h0, w0 = rgb.shape[:2]
        net = cv2.resize(rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
        batch = (net.astype("float32") / 255.0)[None, ...]
        out = self._model.predict(batch, verbose=0)
        if isinstance(out, (list, tuple)):
            out = out[-1]  # deep-supervision: use the final head
        probs = np.asarray(out)[0]
        labels_small = probs.argmax(axis=-1).astype(np.int32)
        label_map = cv2.resize(
            labels_small, (w0, h0), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)
        label_map[label_map >= NUM_CLASSES] = 0
        instances = _instances_from_label_map(label_map)
        return Prediction(
            label_map=label_map,
            instances=instances,
            height=h0,
            width=w0,
            model_name=self.model_name,
            backend=self.backend,
            params={"input_size": [INPUT_HEIGHT, INPUT_WIDTH]},
        )


def build_backend(model_def: dict, checkpoint_path, model_name: str):
    if model_def["backend"] == "yolo":
        return YoloSegBackend(checkpoint_path, model_name=model_name)
    return DenseKerasBackend(checkpoint_path, model_name=model_name)
