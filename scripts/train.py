import argparse
from datetime import datetime
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import numpy as np
import tensorflow as tf
import keras
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmentation_models import (
    ModifiedNestnet,
    ICPRModifiedUnet,
    ICPRUnet,
    TransUNet,
)
from helper_functions import (
    MeanIoUMetric,
    dice_coef,
    dice_coef_loss,
    bce_dice_loss,
    ce_dice_loss,
    ce_dice_boundary_loss,
    multiclass_dice_loss,
    mean_iou,
    iou_score,
)
from project_presets import (
    DEFAULT_SEGMENTATION_MODEL,
    IMAGE_ONLY_SEGMENTATION_MODELS,
    get_segmentation_preset,
)
from protocol_utils import (
    load_json,
    resolve_segmentation_checkpoint,
    summarize_mask_paths,
    validate_bb_map_files,
    validate_disjoint_pair_sets,
    write_json,
)


IMAGE_EXTS = (".jpg", ".jpeg", ".png")
DEFAULT_INPUT_HEIGHT = 512
DEFAULT_INPUT_WIDTH = 1024
DEFAULT_DECODER_BLOCK_TYPE = "upsampling"

SPLITS_DIR = Path(os.environ.get("PBL4_SPLITS_DIR", "data/splits"))
CLASS_MAP_PATH = Path(os.environ.get("PBL4_CLASS_MAP_PATH", "data/splits/class_map.txt"))
OUTPUT_DIR = Path(os.environ.get("PBL4_OUTPUT_DIR", "runs"))
BB_MAPS_ROOT_ENV = os.environ.get("PBL4_BB_MAPS_DIR")
YOLOX_BB_MAPS_ROOT = Path(os.environ.get("PBL4_YOLOX_BB_MAPS_DIR", "data/bb_maps/yolox"))
MASK_RCNN_BB_MAPS_ROOT = Path(os.environ.get("PBL4_MASK_RCNN_BB_MAPS_DIR", "data/bb_maps/mask_rcnn"))
SEED = 13
CLASS_WEIGHTING = "none"
LATEST_ALIAS_NAME = "latest"
LATEST_METADATA_NAME = "latest_run.json"
DEEP_SUPERVISION_HEAD_COUNT = 4
DEFAULT_GPU_DISPLAY_RESERVE_MB = 192
PROCESS_REFRESH_EXIT_CODE = 75


def _env_bool(name):
    value = os.environ.get(name)
    if value is None:
        return None
    return value.lower() in ("1", "true", "yes", "y")


def _env_int(name):
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Low-level segmentation training entrypoint. Common settings come from project presets."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_SEGMENTATION_MODEL,
        choices=[
            "mod_nestnet",
            "icpr_munet",
            "transunet",
            "icpr_unet",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--loss",
        choices=["ce_dice", "ce_dice_boundary", "bce_dice", "dice"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--checkpoint-policy",
        choices=["best", "last"],
        default="best",
        help="Checkpoint used for post-training validation/test evaluation and metadata.",
    )
    parser.add_argument(
        "--eval-test",
        action="store_true",
        help="Evaluate the fixed test split after training. Disabled by default.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=None,
        help="Use Keras mixed_float16 policy to reduce GPU activation memory.",
    )
    parser.add_argument(
        "--bb-source",
        choices=("yolox", "mask_rcnn", "legacy_splits"),
        default=os.environ.get("PBL4_BB_SOURCE"),
        help=(
            "BB-map source for BB-gated models when PBL4_BB_MAPS_DIR is not set. "
            "'yolox' -> data/bb_maps/yolox, 'mask_rcnn' -> data/bb_maps/mask_rcnn, "
            "'legacy_splits' -> current SPLITS_DIR."
        ),
    )
    parser.add_argument(
        "--ds-inference",
        choices=["average", "last", "index"],
        default=None,
        help=(
            "Deep supervision inference output selection used for post-training evaluation: "
            "'average' (mean of all DS heads), 'last' (final DS head), "
            "'index' (use --ds-output-index)."
        ),
    )
    parser.add_argument(
        "--ds-train-head",
        choices=["last", "all", "index"],
        default=os.environ.get("PBL4_DS_TRAIN_HEAD"),
        help=(
            "Training target for mod_nestnet deep-supervision heads. "
            "'last' trains only the final head, 'all' trains every head, "
            "'index' trains --ds-train-output-index only."
        ),
    )
    parser.add_argument(
        "--ds-output-index",
        type=int,
        default=None,
        help="Deep supervision output index when --ds-inference=index.",
    )
    parser.add_argument(
        "--ds-train-output-index",
        type=int,
        default=None,
        help=(
            "Zero-based UNet++ deep-supervision head index when --ds-train-head=index "
            "(0=shallowest, 3=final)."
        ),
    )
    parser.add_argument("--run-dir", default=os.environ.get("PBL4_RUN_DIR"), help=argparse.SUPPRESS)
    parser.add_argument(
        "--process-restart-interval",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-training-backup",
        action="store_true",
        default=_env_bool("PBL4_KEEP_TRAINING_BACKUP") or False,
        help=(
            "Keep the .training_backup/ crash-resume weights after a fold "
            "completes. Off by default: the backup is a full-size duplicate of "
            "the model that is only needed to resume between process restarts."
        ),
    )
    args = parser.parse_args()
    preset = get_segmentation_preset(args.model)
    if args.batch_size is None:
        args.batch_size = preset["batch_size"]
    if args.epochs is None:
        args.epochs = preset["epochs"]
    if args.learning_rate is None:
        args.learning_rate = preset["learning_rate"]
    if args.loss is None:
        args.loss = preset["loss"]
    if args.bb_source is None:
        args.bb_source = preset.get("bb_source", "yolox")
    env_mixed_precision = _env_bool("PBL4_MIXED_PRECISION")
    if args.mixed_precision is None:
        args.mixed_precision = (
            env_mixed_precision
            if env_mixed_precision is not None
            else bool(preset.get("mixed_precision", False))
        )
    cli_ds_train_output_index = args.ds_train_output_index is not None
    if args.ds_train_head is None:
        args.ds_train_head = preset.get("ds_train_head", "last")
    if args.ds_train_head == "index" and args.ds_train_output_index is None:
        args.ds_train_output_index = preset.get("ds_train_output_index")
    elif args.ds_train_head != "index":
        if cli_ds_train_output_index:
            parser.error("--ds-train-output-index can only be used when --ds-train-head=index.")
        args.ds_train_output_index = None
    if args.ds_inference is None:
        args.ds_inference = preset.get("ds_inference", "average")
    args.decoder_filters = preset.get("decoder_filters")
    args.rematerialization = preset.get("rematerialization")
    args.rematerialization_output_size_threshold = preset.get(
        "rematerialization_output_size_threshold",
        1048576,
    )
    args.early_stopping_patience = preset.get("early_stopping_patience")
    if args.process_restart_interval is None:
        args.process_restart_interval = _env_int("PBL4_PROCESS_RESTART_INTERVAL")
    if args.process_restart_interval is None:
        args.process_restart_interval = preset.get("process_restart_interval")
    if args.process_restart_interval is not None and args.process_restart_interval < 1:
        parser.error("--process-restart-interval must be a positive integer.")
    if args.ds_inference == "index" and args.ds_output_index is None:
        parser.error("--ds-output-index is required when --ds-inference=index.")
    if args.ds_inference != "index" and args.ds_output_index is not None:
        parser.error("--ds-output-index can only be used when --ds-inference=index.")
    if args.ds_train_head == "index" and args.ds_train_output_index is None:
        parser.error("--ds-train-output-index is required when --ds-train-head=index.")
    if (
        args.ds_train_output_index is not None
        and not 0 <= args.ds_train_output_index < DEEP_SUPERVISION_HEAD_COUNT
    ):
        parser.error(
            "--ds-train-output-index must be between 0 and "
            f"{DEEP_SUPERVISION_HEAD_COUNT - 1}."
        )
    return args


def resolve_deep_supervision_train_output_index(args):
    if args.model != "mod_nestnet" or args.ds_train_head == "all":
        return None
    if args.ds_train_head == "last":
        return DEEP_SUPERVISION_HEAD_COUNT - 1
    return args.ds_train_output_index


def _resolve_default_bb_maps_root(splits_dir: Path, bb_source: str) -> Path:
    if bb_source == "legacy_splits":
        return splits_dir
    source_root = MASK_RCNN_BB_MAPS_ROOT if bb_source == "mask_rcnn" else YOLOX_BB_MAPS_ROOT
    # CV folds use SPLITS_DIR=data/splits/folds/fold_<k>; map to source-root fold layout.
    if splits_dir.name.startswith("fold_") and splits_dir.parent.name == "folds":
        return source_root / "folds" / splits_dir.name
    return source_root


def _allocate_run_dir(run_root: Path, run_name: str) -> Path:
    stem = f"{run_name}{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    candidate = run_root / stem
    suffix = 1
    while candidate.exists():
        candidate = run_root / f"{stem}_{suffix}"
        suffix += 1
    return candidate


def _clear_latest_compat_files(run_root: Path) -> None:
    for path in run_root.iterdir():
        if not path.is_file():
            continue
        if path.name == LATEST_METADATA_NAME:
            continue
        if path.suffix in {".keras", ".json"}:
            path.unlink()


def _publish_latest_run(run_root: Path, run_dir: Path) -> None:
    run_root = Path(run_root)
    run_dir = Path(run_dir)

    latest_alias = run_root / LATEST_ALIAS_NAME
    if latest_alias.exists() or latest_alias.is_symlink():
        if latest_alias.is_symlink() or latest_alias.is_file():
            latest_alias.unlink()
        else:
            shutil.rmtree(latest_alias)
    try:
        latest_alias.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        # Symlink can fail on some systems; metadata + copied files still provide
        # a stable "latest" entrypoint.
        pass

    _clear_latest_compat_files(run_root)
    # Only copy the lightweight .json artifacts (metadata + per-class metrics)
    # to the run-root for legacy tooling that reads from a stable path. We do
    # NOT copy .keras checkpoints — that previously doubled the on-disk size
    # of every fold and inflated the output-zip by gigabytes. The `latest`
    # symlink + timestamped subdirs are enough for resolve_segmentation_checkpoint
    # to find best.keras / last.keras.
    for path in run_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix != ".json":
            continue
        shutil.copy2(path, run_root / path.name)

    write_json(
        run_root / LATEST_METADATA_NAME,
        {
            "latest_run_name": run_dir.name,
            "latest_run_dir": str(run_dir),
            "relative_run_dir": run_dir.name,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


class ProcessRefreshRequested(Exception):
    def __init__(self, completed_epoch):
        super().__init__(f"process refresh requested after epoch {completed_epoch}")
        self.completed_epoch = completed_epoch


def _format_log_value(value):
    try:
        value = np.asarray(value)
        if value.size != 1:
            return None
        value = float(value.reshape(()))
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return str(value)
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.4e}"
    return f"{value:.4f}"


def _format_epoch_logs(logs):
    logs = logs or {}
    preferred = (
        "loss",
        "acc",
        "mean_iou",
        "dice_coef",
        "iou_score",
        "val_loss",
        "val_acc",
        "val_mean_iou",
        "val_dice_coef",
        "val_iou_score",
        "learning_rate",
    )
    ordered_keys = [key for key in preferred if key in logs]
    ordered_keys.extend(key for key in logs if key not in set(ordered_keys))

    parts = []
    for key in ordered_keys:
        formatted = _format_log_value(logs[key])
        if formatted is not None:
            parts.append(f"{key}: {formatted}")
    return " - ".join(parts)


class ProcessRefreshCallback(keras.callbacks.Callback):
    def __init__(self, interval, total_epochs, state_path):
        super().__init__()
        self.interval = interval
        self.total_epochs = total_epochs
        self.state_path = Path(state_path)

    def on_epoch_end(self, epoch, logs=None):
        if not self.interval:
            return
        completed_epoch = epoch + 1
        if completed_epoch >= self.total_epochs:
            return
        if completed_epoch % self.interval != 0:
            return

        state = load_json(self.state_path, default={}) or {}
        if state.get("stop_training_requested"):
            return
        state["refresh_requested_after_epoch"] = int(completed_epoch)
        state["refresh_reason"] = "gpu_memory_pool_reset"
        write_json(self.state_path, state)
        formatted_logs = _format_epoch_logs(logs)
        if formatted_logs:
            print(f"Epoch {completed_epoch} completed before refresh - {formatted_logs}")
        raise ProcessRefreshRequested(completed_epoch)


class TrainingStateCallback(keras.callbacks.Callback):
    def __init__(self, state_path):
        super().__init__()
        self.state_path = Path(state_path)
        self.state = load_json(self.state_path, default={}) or {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.state = load_json(self.state_path, default={}) or {}
        completed_epoch = epoch + 1
        self.state["last_completed_epoch"] = int(completed_epoch)

        val_loss = logs.get("val_loss")
        if val_loss is not None:
            best_val_loss = self.state.get("best_val_loss")
            if best_val_loss is None or float(val_loss) < float(best_val_loss):
                self.state["best_val_loss"] = float(val_loss)
                self.state["best_epoch"] = int(completed_epoch)

        write_json(self.state_path, self.state)


class PersistentEarlyStopping(keras.callbacks.Callback):
    def __init__(self, monitor, patience, state_path, min_delta=0.0):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.state_path = Path(state_path)
        self.min_delta = float(min_delta)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None or self.patience is None:
            return

        completed_epoch = epoch + 1
        current = float(current)
        state = load_json(self.state_path, default={}) or {}
        best = state.get("early_stopping_best")
        if best is None and self.monitor == "val_loss":
            best = state.get("best_val_loss")
        wait = int(state.get("early_stopping_wait", 0))

        if best is None or current < float(best) - self.min_delta:
            best = current
            wait = 0
            state["early_stopping_best_epoch"] = int(completed_epoch)
        else:
            wait += 1

        state["early_stopping_monitor"] = self.monitor
        state["early_stopping_patience"] = int(self.patience)
        state["early_stopping_best"] = float(best)
        state["early_stopping_wait"] = int(wait)

        if wait >= self.patience:
            state["early_stopping_stopped_epoch"] = int(completed_epoch)
            state["stop_training_requested"] = True
            print(
                f"Early stopping triggered at epoch {completed_epoch}: "
                f"{self.monitor} has not improved for {self.patience} epochs."
            )
            self.model.stop_training = True

        write_json(self.state_path, state)


def _query_gpu_total_memory_mb():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    totals = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            totals.append(int(line))
        except ValueError:
            continue
    return totals


def configure_tensorflow_memory():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return

    explicit_limit_mb = _env_int("PBL4_GPU_MEMORY_LIMIT_MB")
    display_reserve_mb = _env_int("PBL4_GPU_DISPLAY_RESERVE_MB")
    if display_reserve_mb is None:
        display_reserve_mb = DEFAULT_GPU_DISPLAY_RESERVE_MB

    total_memory_mb = _query_gpu_total_memory_mb()
    configured_fixed_limit = False

    for idx, gpu in enumerate(gpus):
        memory_limit_mb = explicit_limit_mb
        if memory_limit_mb is None and idx < len(total_memory_mb):
            memory_limit_mb = total_memory_mb[idx] - display_reserve_mb

        if memory_limit_mb is not None and memory_limit_mb > 0:
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
                )
                print(
                    "TensorFlow GPU memory limit set:",
                    f"gpu={idx}",
                    f"limit={memory_limit_mb} MiB",
                    f"display_reserve={display_reserve_mb} MiB",
                )
                configured_fixed_limit = True
                continue
            except (RuntimeError, ValueError) as exc:
                print(
                    "warning: failed to set TensorFlow GPU memory limit; "
                    f"falling back to memory growth ({exc})"
                )

        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if not configured_fixed_limit:
        print("TensorFlow GPU memory growth enabled.")


def configure_mixed_precision(enabled):
    if enabled:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled: mixed_float16")


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
    if max_id == 0:
        return None
    return max_id + 1


def load_class_map(class_map_path):
    path = Path(class_map_path)
    if not path.exists():
        return {}
    mapping = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 1:
            name = parts[0]
            idx = int(parts[0])
        else:
            name = parts[0]
            idx = int(parts[-1])
        mapping[idx] = name
    return mapping


def compute_class_weights(mask_paths, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for path in mask_paths:
        mask = np.array(Image.open(path))
        counts += np.bincount(mask.reshape(-1), minlength=num_classes)

    counts = counts.astype(np.float64)
    if num_classes > 1:
        non_bg = counts[1:]
        non_zero = non_bg[non_bg > 0]
        if non_zero.size == 0:
            return np.ones(num_classes, dtype=np.float32)
        median = np.median(non_zero)
    else:
        median = counts[0] if counts[0] > 0 else 1.0

    weights = median / np.maximum(counts, 1.0)
    return weights.astype(np.float32)


def make_weighted_losses(class_weights, num_classes):
    weights = tf.constant(class_weights, dtype=tf.float32)

    def weighted_sparse_ce(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        w = tf.gather(weights, y_true)
        return tf.reduce_mean(ce * w)

    def weighted_dice(y_true, y_pred, smooth=1.0):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_classes, dtype=tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        axes = (0, 1, 2)
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        denom = tf.reduce_sum(y_true + y_pred, axis=axes)
        dice = (2.0 * intersection + smooth) / (denom + smooth)
        return 1.0 - tf.reduce_sum(dice * weights) / tf.reduce_sum(weights)

    def weighted_ce_dice(y_true, y_pred):
        return weighted_sparse_ce(y_true, y_pred) + weighted_dice(y_true, y_pred)

    return weighted_ce_dice, weighted_dice


def list_pairs(images_dir, masks_dir, bb_maps_dir=None, bb_ext=".npz"):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)
    masks = {p.stem: p for p in masks_dir.glob("*.png")}
    bb_maps = {}
    if bb_maps_dir is not None:
        bb_maps_dir = Path(bb_maps_dir)
        bb_maps = {p.stem: p for p in bb_maps_dir.glob(f"*{bb_ext}")}
    image_paths = []
    for ext in IMAGE_EXTS:
        image_paths.extend(images_dir.glob(f"*{ext}"))
    pairs = []
    missing_bb = []
    for img in sorted(image_paths):
        mask = masks.get(img.stem)
        if mask is not None:
            if bb_maps_dir is None:
                pairs.append((str(img), str(mask)))
            else:
                bb_path = bb_maps.get(img.stem)
                if bb_path is None:
                    missing_bb.append(img.name)
                    continue
                pairs.append((str(img), str(mask), str(bb_path)))
    if bb_maps_dir is not None and missing_bb:
        sample = ", ".join(missing_bb[:5])
        raise SystemExit(f"Missing {len(missing_bb)} BB map files in {bb_maps_dir} (e.g. {sample})")
    return pairs


def _decode_image(path):
    data = tf.io.read_file(path)
    img = tf.image.decode_image(data, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def _decode_mask(path):
    data = tf.io.read_file(path)
    mask = tf.image.decode_png(data, channels=1)
    return mask


def _load_bb_map_np(path):
    data = np.load(path.decode("utf-8"))
    if isinstance(data, np.lib.npyio.NpzFile):
        if "bb" in data:
            array = data["bb"]
        else:
            array = data[data.files[0]]
    else:
        array = data
    return array.astype(np.float32)


def load_bb_map(path, size, bb_channels):
    bb_map = tf.numpy_function(_load_bb_map_np, [path], tf.float32)
    bb_map.set_shape([None, None, bb_channels])
    if size is not None:
        bb_map = tf.image.resize(bb_map, size, method="nearest")
    bb_map = tf.cast(bb_map > 0.5, tf.float32)
    return bb_map


def load_pair(img_path, mask_path, size, num_classes, preprocess_fn, scale_to_255):
    image = _decode_image(img_path)
    mask = _decode_mask(mask_path)

    if size is not None:
        image = tf.image.resize(image, size, method="bilinear")
        mask = tf.image.resize(mask, size, method="nearest")

    if preprocess_fn is not None:
        if scale_to_255:
            image = image * 255.0
        image = preprocess_fn(image)

    if num_classes == 1:
        mask = tf.cast(mask > 0, tf.float32)
    else:
        mask = tf.cast(tf.squeeze(mask, axis=-1), tf.int32)

    return image, mask


def load_pair_with_bb(
    img_path,
    mask_path,
    bb_path,
    size,
    num_classes,
    preprocess_fn,
    scale_to_255,
    bb_channels,
):
    image, mask = load_pair(img_path, mask_path, size, num_classes, preprocess_fn, scale_to_255)
    bb_map = load_bb_map(bb_path, size, bb_channels)
    return (image, bb_map), mask


def _is_multi_output_model(model):
    return len(getattr(model, "outputs", [])) > 1


def _make_training_targets_for_outputs(dataset, output_names):
    output_names = tuple(output_names)

    def _map_to_output_targets(inputs, mask):
        return inputs, {name: mask for name in output_names}

    return dataset.map(_map_to_output_targets, num_parallel_calls=tf.data.AUTOTUNE)


def _select_prediction_tensor(preds):
    if isinstance(preds, dict):
        if not preds:
            raise ValueError("Model returned an empty prediction dictionary.")
        pred_list = list(preds.values())
        return np.mean(np.stack(pred_list, axis=0), axis=0)
    if isinstance(preds, (list, tuple)):
        if not preds:
            raise ValueError("Model returned an empty prediction list.")
        return np.mean(np.stack(preds, axis=0), axis=0)
    return preds


def _build_inference_model(model, ds_inference, ds_output_index):
    if not _is_multi_output_model(model):
        return model, None
    output_names = list(model.output_names)
    if ds_inference == "average":
        inference_output = keras.layers.Average(name="deep_supervision_average")(model.outputs)
        inference_name = "deep_supervision_average"
    elif ds_inference == "last":
        inference_output = model.outputs[-1]
        inference_name = output_names[-1]
    else:
        if ds_output_index < 0 or ds_output_index >= len(model.outputs):
            raise ValueError(
                f"Invalid --ds-output-index={ds_output_index} for {len(model.outputs)} outputs."
            )
        inference_output = model.outputs[ds_output_index]
        inference_name = output_names[ds_output_index]
    inference_model = keras.Model(model.inputs, inference_output, name=f"{model.name}_inference")
    return inference_model, inference_name



def _safe_divide(numerator, denominator):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )


def compute_confusion_matrix(dataset, model, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, masks in dataset:
        preds = _select_prediction_tensor(model.predict(images, verbose=0))
        if isinstance(masks, dict):
            masks = list(masks.values())[-1]
        elif isinstance(masks, (list, tuple)):
            masks = masks[-1]
        if preds.shape[-1] == 1:
            pred_labels = (preds[..., 0] > 0.5).astype(np.int32)
            true_labels = masks.numpy().astype(np.int32)
        else:
            pred_labels = np.argmax(preds, axis=-1).astype(np.int32)
            true_labels = masks.numpy().astype(np.int32)
        conf = tf.math.confusion_matrix(
            true_labels.reshape(-1),
            pred_labels.reshape(-1),
            num_classes=num_classes,
        ).numpy()
        confusion += conf
    return confusion


def per_class_metrics_from_confusion(confusion):
    true_positives = np.diag(confusion)
    false_positives = confusion.sum(axis=0) - true_positives
    false_negatives = confusion.sum(axis=1) - true_positives

    iou = _safe_divide(true_positives, true_positives + false_positives + false_negatives)
    dice = _safe_divide(
        2.0 * true_positives,
        2.0 * true_positives + false_positives + false_negatives,
    )
    support = confusion.sum(axis=1)
    return iou, dice, support


_QUADRANT_DEFINITIONS = (
    (1, 8, "upper_left", "molar_to_incisor"),
    (9, 16, "upper_right", "incisor_to_molar"),
    (17, 24, "lower_right", "molar_to_incisor"),
    (25, 32, "lower_left", "incisor_to_molar"),
)

_POSITION_LABELS = {
    1: "central_incisor",
    2: "lateral_incisor",
    3: "canine",
    4: "first_premolar",
    5: "second_premolar",
    6: "first_molar",
    7: "second_molar",
    8: "third_molar",
}


def _class_id_to_position(class_id):
    if class_id <= 0:
        return None
    for start, end, _, direction in _QUADRANT_DEFINITIONS:
        if start <= class_id <= end:
            offset = class_id - start
            if direction == "incisor_to_molar":
                return offset + 1
            return 8 - offset
    return None


def _class_id_to_quadrant(class_id):
    if class_id <= 0:
        return None
    for start, end, quadrant, _ in _QUADRANT_DEFINITIONS:
        if start <= class_id <= end:
            return quadrant
    return None


def _class_id_to_tooth_type(class_id):
    position = _class_id_to_position(class_id)
    if position is None:
        return None
    if position in (1, 2):
        return "incisor"
    if position == 3:
        return "canine"
    if position in (4, 5):
        return "premolar"
    return "molar"


def _summarize_groups(groups, iou, dice, support):
    rows = []
    for name, class_ids in groups.items():
        values_iou = [iou[class_id] for class_id in class_ids]
        values_dice = [dice[class_id] for class_id in class_ids]
        pixels = int(sum(support[class_id] for class_id in class_ids))
        rows.append(
            {
                "group": name,
                "class_ids": class_ids,
                "iou_mean": float(np.mean(values_iou)),
                "iou_std": float(np.std(values_iou)),
                "dice_mean": float(np.mean(values_dice)),
                "dice_std": float(np.std(values_dice)),
                "pixels": pixels,
            }
        )
    return rows


def report_per_class_metrics(name, confusion, class_names, out_dir):
    iou, dice, support = per_class_metrics_from_confusion(confusion)
    rows = []
    for class_id, (iou_val, dice_val, count) in enumerate(zip(iou, dice, support)):
        class_name = class_names.get(class_id, str(class_id))
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "iou": float(iou_val),
                "dice": float(dice_val),
                "pixels": int(count),
            }
        )
    rows.sort(key=lambda r: r["iou"])
    out_path = Path(out_dir) / f"per_class_metrics_{name}.json"
    out_path.write_text(json.dumps(rows, indent=2))

    worst = [r for r in rows if r["class_id"] != 0][:5]
    print(f"Per-class {name} metrics (worst IoU):")
    for row in worst:
        print(
            f"  class {row['class_id']} ({row['class_name']}): "
            f"IoU={row['iou']:.4f} Dice={row['dice']:.4f} pixels={row['pixels']}"
        )
    return iou, dice, support


def report_group_metrics(name, iou, dice, support, out_dir, num_classes):
    if num_classes - 1 != 32:
        print("Skipping per-position metrics (expected 32 tooth classes).")
        return

    position_groups = {}
    for class_id in range(1, num_classes):
        position = _class_id_to_position(class_id)
        if position is None:
            continue
        position_groups.setdefault(position, []).append(class_id)

    tooth_groups = {"incisor": [], "canine": [], "premolar": [], "molar": []}
    for class_id in range(1, num_classes):
        tooth_type = _class_id_to_tooth_type(class_id)
        if tooth_type is not None:
            tooth_groups[tooth_type].append(class_id)

    labeled_positions = {}
    for pos, class_ids in position_groups.items():
        label = _POSITION_LABELS.get(pos, f"pos_{pos}")
        labeled_positions[f"{pos}_{label}"] = class_ids

    position_rows = _summarize_groups(labeled_positions, iou, dice, support)
    position_rows.sort(key=lambda row: row["group"])
    (Path(out_dir) / f"per_position_metrics_{name}.json").write_text(
        json.dumps(position_rows, indent=2)
    )

    tooth_rows = _summarize_groups(tooth_groups, iou, dice, support)
    (Path(out_dir) / f"per_tooth_type_metrics_{name}.json").write_text(
        json.dumps(tooth_rows, indent=2)
    )

    quadrant_groups = {}
    for class_id in range(1, num_classes):
        quadrant = _class_id_to_quadrant(class_id)
        if quadrant is None:
            continue
        quadrant_groups.setdefault(quadrant, []).append(class_id)
    quadrant_rows = _summarize_groups(quadrant_groups, iou, dice, support)
    quadrant_rows.sort(key=lambda row: row["group"])
    (Path(out_dir) / f"per_quadrant_metrics_{name}.json").write_text(
        json.dumps(quadrant_rows, indent=2)
    )

    print(f"Per-position {name} metrics saved to per_position_metrics_{name}.json")
    print(f"Per-tooth-type {name} metrics saved to per_tooth_type_metrics_{name}.json")
    print(f"Per-quadrant {name} metrics saved to per_quadrant_metrics_{name}.json")


def make_dataset(
    pairs,
    size,
    num_classes,
    batch_size,
    shuffle,
    augment,
    preprocess_fn,
    scale_to_255,
    bb_channels=None,
):
    img_paths = [p[0] for p in pairs]
    mask_paths = [p[1] for p in pairs]
    if bb_channels is None:
        ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    else:
        bb_paths = [p[2] for p in pairs]
        ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, bb_paths))
    if shuffle:
        ds = ds.shuffle(len(pairs), reshuffle_each_iteration=True)
    if bb_channels is None:
        ds = ds.map(
            lambda x, y: load_pair(x, y, size, num_classes, preprocess_fn, scale_to_255),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        ds = ds.map(
            lambda x, y, z: load_pair_with_bb(
                x, y, z, size, num_classes, preprocess_fn, scale_to_255, bb_channels
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(args, num_classes, input_shape, bb_channels):
    activation = "sigmoid" if num_classes == 1 else "softmax"
    classes = 1 if num_classes == 1 else num_classes

    if args.model == "mod_nestnet":
        model_kwargs = dict(
            input_shape=input_shape,
            decoder_block_type=DEFAULT_DECODER_BLOCK_TYPE,
            decoder_filters=tuple(args.decoder_filters) if args.decoder_filters else None,
            classes=classes,
            activation=activation,
            bb_channels=bb_channels,
            deep_supervision=args.ds_train_head == "all",
            deep_supervision_output_index=resolve_deep_supervision_train_output_index(args),
        )
        if args.rematerialization:
            print(
                "Rematerialization enabled:",
                args.rematerialization,
                f"threshold={args.rematerialization_output_size_threshold}",
            )
            with keras.RematScope(
                mode=args.rematerialization,
                output_size_threshold=args.rematerialization_output_size_threshold,
            ):
                return ModifiedNestnet(**model_kwargs)
        return ModifiedNestnet(**model_kwargs)
    if args.model == "icpr_munet":
        return ICPRModifiedUnet(
            input_shape=input_shape,
            classes=classes,
            activation=activation,
            bb_channels=bb_channels,
        )
    if args.model == "transunet":
        return TransUNet(
            input_shape=input_shape,
            classes=classes,
            activation=activation,
        )
    if args.model == "icpr_unet":
        return ICPRUnet(
            input_shape=input_shape,
            classes=classes,
            activation=activation,
        )
    raise ValueError(f"Unsupported model {args.model}")


def main():
    args = parse_args()
    configure_tensorflow_memory()
    configure_mixed_precision(args.mixed_precision)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    height = DEFAULT_INPUT_HEIGHT
    width = DEFAULT_INPUT_WIDTH

    input_shape = (height, width, 3)
    size = (height, width)

    num_classes = infer_num_classes(CLASS_MAP_PATH) or 1
    preprocess_fn = None
    scale_to_255 = False
    image_only = args.model in IMAGE_ONLY_SEGMENTATION_MODELS
    if image_only:
        bb_maps_root = None
        bb_channels = None
    else:
        bb_maps_root = (
            Path(BB_MAPS_ROOT_ENV)
            if BB_MAPS_ROOT_ENV
            else _resolve_default_bb_maps_root(SPLITS_DIR, args.bb_source)
        )
        if not bb_maps_root.exists():
            raise SystemExit(f"BB maps directory not found: {bb_maps_root}")
        bb_channels = num_classes

    base = SPLITS_DIR
    train_pairs = list_pairs(
        base / "train" / "img",
        base / "train" / "masks_semantic",
        bb_maps_dir=bb_maps_root / "train" / "bb_maps" if bb_maps_root else None,
    )
    val_pairs = list_pairs(
        base / "val" / "img",
        base / "val" / "masks_semantic",
        bb_maps_dir=bb_maps_root / "val" / "bb_maps" if bb_maps_root else None,
    )
    test_pairs = []
    if args.eval_test:
        test_pairs = list_pairs(
            base / "test" / "img",
            base / "test" / "masks_semantic",
            bb_maps_dir=bb_maps_root / "test" / "bb_maps" if bb_maps_root else None,
        )
    if not train_pairs:
        raise SystemExit("No training image/mask pairs found.")
    if not val_pairs:
        raise SystemExit("No validation image/mask pairs found.")
    validate_disjoint_pair_sets(
        {
            "train": train_pairs,
            "val": val_pairs,
            **({"test": test_pairs} if test_pairs else {}),
        }
    )
    print(
        "Dataset split:",
        f"splits_dir={base}",
        f"train={len(train_pairs)}",
        f"val={len(val_pairs)}",
        f"test={len(test_pairs)}",
    )
    if bb_maps_root is not None:
        print(f"BB maps split root: {bb_maps_root}")

    train_mask_stats = summarize_mask_paths(
        [pair[1] for pair in train_pairs], num_classes, subset_name="train"
    )
    val_mask_stats = summarize_mask_paths(
        [pair[1] for pair in val_pairs], num_classes, subset_name="val"
    )
    test_mask_stats = None
    if test_pairs:
        test_mask_stats = summarize_mask_paths(
            [pair[1] for pair in test_pairs], num_classes, subset_name="test"
        )

    train_bb_stats = None
    val_bb_stats = None
    test_bb_stats = None
    if bb_channels is not None:
        train_bb_stats = validate_bb_map_files(
            [pair[2] for pair in train_pairs],
            expected_height=height,
            expected_width=width,
            expected_channels=bb_channels,
            subset_name="train",
        )
        val_bb_stats = validate_bb_map_files(
            [pair[2] for pair in val_pairs],
            expected_height=height,
            expected_width=width,
            expected_channels=bb_channels,
            subset_name="val",
        )
        if test_pairs:
            test_bb_stats = validate_bb_map_files(
                [pair[2] for pair in test_pairs],
                expected_height=height,
                expected_width=width,
                expected_channels=bb_channels,
                subset_name="test",
            )

    class_weights = None
    if num_classes > 1 and CLASS_WEIGHTING != "none":
        mask_paths = [pair[1] for pair in train_pairs]
        class_weights = compute_class_weights(mask_paths, num_classes)
        print(
            "Class weighting enabled:",
            f"min={class_weights.min():.4f}",
            f"max={class_weights.max():.4f}",
        )

    train_ds = make_dataset(
        train_pairs,
        size=size,
        num_classes=num_classes,
        batch_size=args.batch_size,
        shuffle=True,
        augment=False,
        preprocess_fn=preprocess_fn,
        scale_to_255=scale_to_255,
        bb_channels=bb_channels,
    )
    val_ds = make_dataset(
        val_pairs,
        size=size,
        num_classes=num_classes,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
        preprocess_fn=preprocess_fn,
        scale_to_255=scale_to_255,
        bb_channels=bb_channels,
    )

    model = build_model(args, num_classes, input_shape, bb_channels)
    is_deep_supervision = _is_multi_output_model(model)
    deep_supervision_outputs = list(model.output_names) if is_deep_supervision else []
    deep_supervision_final_output = (
        deep_supervision_outputs[-1] if deep_supervision_outputs else None
    )
    deep_supervision_train_output_index = resolve_deep_supervision_train_output_index(args)
    deep_supervision_train_outputs = (
        list(model.output_names) if args.model == "mod_nestnet" else None
    )

    train_fit_ds = train_ds
    val_fit_ds = val_ds
    if is_deep_supervision:
        train_fit_ds = _make_training_targets_for_outputs(train_ds, deep_supervision_outputs)
        val_fit_ds = _make_training_targets_for_outputs(val_ds, deep_supervision_outputs)

    optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)
    if num_classes == 1:
        if args.loss == "dice":
            loss = dice_coef_loss
        elif args.loss == "bce_dice":
            loss = bce_dice_loss
        elif args.loss == "ce_dice_boundary":
            loss = ce_dice_boundary_loss
        else:
            loss = ce_dice_loss
        metrics = [dice_coef, mean_iou, iou_score]
    else:
        if class_weights is not None:
            weighted_ce_dice, weighted_dice = make_weighted_losses(class_weights, num_classes)
            loss = weighted_dice if args.loss == "dice" else weighted_ce_dice
        else:
            if args.loss == "dice":
                loss = multiclass_dice_loss
            elif args.loss == "ce_dice_boundary":
                loss = ce_dice_boundary_loss
            else:
                loss = ce_dice_loss
        metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="acc"),
            MeanIoUMetric(num_classes=num_classes, name="mean_iou"),
        ]

    if is_deep_supervision:
        loss_by_output = {name: loss for name in deep_supervision_outputs}
        output_weight = 1.0 / float(len(deep_supervision_outputs))
        loss_weights = {name: output_weight for name in deep_supervision_outputs}
        model.compile(
            optimizer=optimizer,
            loss=loss_by_output,
            loss_weights=loss_weights,
            metrics={deep_supervision_final_output: metrics},
            jit_compile=False,
        )
    else:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)

    run_name = args.model
    run_root = OUTPUT_DIR / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.run_dir) if args.run_dir else _allocate_run_dir(run_root, run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving run artifacts to {out_dir}")

    training_state_path = out_dir / "training_state.json"
    training_state = load_json(training_state_path, default={}) or {}
    backup_dir = out_dir / ".training_backup"
    best_val_loss = training_state.get("best_val_loss")

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
            initial_value_threshold=best_val_loss,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=15,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=str(out_dir / "logs")),
    ]
    if args.model in ("mod_nestnet", "transunet") and args.early_stopping_patience:
        callbacks.append(
            PersistentEarlyStopping(
                monitor="val_loss",
                patience=int(args.early_stopping_patience),
                state_path=training_state_path,
            )
        )
    callbacks.append(TrainingStateCallback(training_state_path))
    if args.process_restart_interval:
        callbacks.append(
            keras.callbacks.BackupAndRestore(
                backup_dir=str(backup_dir),
                save_freq="epoch",
                # Delete the backup once the fold finishes normally. Intermediate
                # process refreshes raise ProcessRefreshRequested out of fit() so
                # on_train_end never fires and the backup survives for resume;
                # only genuine completion (final epoch) triggers deletion.
                delete_checkpoint=not args.keep_training_backup,
            )
        )
        callbacks.append(
            ProcessRefreshCallback(
                interval=args.process_restart_interval,
                total_epochs=args.epochs,
                state_path=training_state_path,
            )
        )

    try:
        model.fit(
            train_fit_ds,
            validation_data=val_fit_ds if val_pairs else None,
            epochs=args.epochs,
            callbacks=callbacks,
        )
    except ProcessRefreshRequested as exc:
        print(
            "Refreshing TensorFlow process after epoch "
            f"{exc.completed_epoch}; relaunching frees the CUDA memory pool."
        )
        sys.exit(PROCESS_REFRESH_EXIT_CODE)

    model.save(out_dir / "last.keras")
    # Fold finished: drop the crash-resume backup (a full-size weight duplicate)
    # unless explicitly kept. Reaching here means fit() completed without a
    # refresh, so the backup is no longer needed.
    if not args.keep_training_backup and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)
    eval_checkpoint, checkpoint_info = resolve_segmentation_checkpoint(
        out_dir, checkpoint_policy=args.checkpoint_policy
    )
    print(f"Using checkpoint for post-training evaluation: {eval_checkpoint}")
    loaded_eval_model = keras.models.load_model(eval_checkpoint, compile=False)
    eval_model, eval_output_name = _build_inference_model(
        loaded_eval_model,
        ds_inference=args.ds_inference,
        ds_output_index=args.ds_output_index,
    )
    if eval_output_name is not None:
        print(f"Deep supervision enabled; evaluation output: {eval_output_name}")
    eval_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics=metrics,
    )

    run_metadata = {
        "model": args.model,
        "encoder": "icpr",
        "loss": args.loss,
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "mixed_precision": bool(args.mixed_precision),
        "decoder_filters": list(args.decoder_filters) if args.decoder_filters else None,
        "process_restart_interval": (
            int(args.process_restart_interval) if args.process_restart_interval else None
        ),
        "early_stopping_patience": (
            int(args.early_stopping_patience) if args.early_stopping_patience else None
        ),
        "rematerialization": args.rematerialization,
        "rematerialization_output_size_threshold": (
            int(args.rematerialization_output_size_threshold)
            if args.rematerialization_output_size_threshold is not None
            else None
        ),
        "run_name": run_name,
        "run_root": str(run_root),
        "run_dir": str(out_dir),
        "splits_dir": str(base),
        "class_map_path": str(CLASS_MAP_PATH),
        "bb_maps_root": str(bb_maps_root) if bb_maps_root else None,
        "bb_source": args.bb_source if bb_maps_root else None,
        "num_classes": int(num_classes),
        "input_shape": list(input_shape),
        "seed": int(SEED),
        "deep_supervision": bool(is_deep_supervision),
        "deep_supervision_train_strategy": (
            args.ds_train_head if args.model == "mod_nestnet" else None
        ),
        "deep_supervision_train_output_index": deep_supervision_train_output_index,
        "deep_supervision_train_outputs": deep_supervision_train_outputs,
        "deep_supervision_outputs": deep_supervision_outputs if is_deep_supervision else None,
        "deep_supervision_final_output": deep_supervision_final_output,
        "deep_supervision_eval_strategy": args.ds_inference if is_deep_supervision else None,
        "deep_supervision_eval_output_index": (
            args.ds_output_index if (is_deep_supervision and args.ds_inference == "index") else None
        ),
        "deep_supervision_eval_output": eval_output_name if is_deep_supervision else None,
        "checkpoint": {
            **checkpoint_info,
            "path": str(eval_checkpoint),
        },
        "train": {
            "samples": len(train_pairs),
            **train_mask_stats,
            **({"bb_maps": train_bb_stats} if train_bb_stats is not None else {}),
        },
        "val": {
            "samples": len(val_pairs),
            **val_mask_stats,
            **({"bb_maps": val_bb_stats} if val_bb_stats is not None else {}),
        },
        "test": None,
        "evaluated_test_during_training": bool(args.eval_test),
    }
    if test_pairs and test_mask_stats is not None:
        run_metadata["test"] = {
            "samples": len(test_pairs),
            **test_mask_stats,
            **({"bb_maps": test_bb_stats} if test_bb_stats is not None else {}),
        }
    write_json(out_dir / "run_metadata.json", run_metadata)

    if num_classes > 1:
        class_names = load_class_map(CLASS_MAP_PATH)
        class_names.setdefault(0, "background")
        val_confusion = compute_confusion_matrix(val_ds, eval_model, num_classes)
        val_iou, val_dice, val_support = report_per_class_metrics(
            "val", val_confusion, class_names, out_dir
        )
        report_group_metrics("val", val_iou, val_dice, val_support, out_dir, num_classes)

    if test_pairs:
        test_ds = make_dataset(
            test_pairs,
            size=size,
            num_classes=num_classes,
            batch_size=args.batch_size,
            shuffle=False,
            augment=False,
            preprocess_fn=preprocess_fn,
            scale_to_255=scale_to_255,
            bb_channels=bb_channels,
        )
        results = eval_model.evaluate(test_ds, verbose=1)
        metric_names = eval_model.metrics_names
        report = dict(zip(metric_names, results))
        print("Test metrics:", report)
        write_json(out_dir / "test_metrics.json", report)
        if num_classes > 1:
            test_confusion = compute_confusion_matrix(test_ds, eval_model, num_classes)
            test_iou, test_dice, test_support = report_per_class_metrics(
                "test", test_confusion, class_names, out_dir
            )
            report_group_metrics("test", test_iou, test_dice, test_support, out_dir, num_classes)

    _publish_latest_run(run_root, out_dir)
    print(
        f"Updated latest run pointer at {run_root / LATEST_ALIAS_NAME} "
        f"and compatibility files under {run_root}"
    )


if __name__ == "__main__":
    main()
