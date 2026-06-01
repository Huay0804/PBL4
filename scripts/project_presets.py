from copy import deepcopy


DEFAULT_SEGMENTATION_MODEL = "mod_nestnet"

SEGMENTATION_PRESETS = {
    "mod_nestnet": {
        "batch_size": 1,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
        "bb_source": "yolox",
        "mixed_precision": True,
        "ds_train_head": "index",
        "ds_train_output_index": 2,
        "ds_inference": "last",
        "decoder_filters": (512, 256, 128, 64),
        "process_restart_interval": 5,
        "early_stopping_patience": 4,
    },
    "icpr_munet": {
        "batch_size": 1,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
        "bb_source": "mask_rcnn",
        "mixed_precision": False,
    },
    "transunet": {
        "batch_size": 1,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
        "mixed_precision": True,
        "process_restart_interval": 5,
        "early_stopping_patience": 4,
    },
}

# Image-only segmentation models do not consume bounding-box prior maps.
IMAGE_ONLY_SEGMENTATION_MODELS = ("transunet",)

MASK_RCNN_PRESET = {
    "epochs": 80,
    "batch_size": 1,
    "learning_rate": 0.001,
}

YOLOX_PRESET = {
    "experiment_name": "yolox_teeth_s",
    "epochs": 80,
    "batch_size": 4,
    "learning_rate": 0.001,
    "input_height": 512,
    "input_width": 1024,
    "num_workers": 4,
    "conf_threshold": 0.01,
    "nms_threshold": 0.55,
    "weights": "coco",
}


# Ultralytics YOLO instance-segmentation baselines run under the SAME protocol
# as the dense segmenters (4-fold CV + fixed-test eval, per-class/position/
# tooth-type IoU+Dice). They predict per-instance masks, which evaluate_yolo_seg
# rasterizes back into a 33-class label map before scoring, so the resulting
# test_summary.json is directly comparable to TransUNet's.
#
# `weights_template` is formatted with the chosen size letter (n/s/m/l/x).
# yolo11 ships COCO-pretrained -seg weights (the proven baseline); yolo26 is the
# Jan-2026 SOTA "latest" comparison.
YOLO_SEG_MODELS = {
    "yolo11": {
        "weights_template": "yolo11{size}-seg.pt",
        "run_name": "yolo11_seg",
    },
    "yolo26": {
        "weights_template": "yolo26{size}-seg.pt",
        "run_name": "yolo26_seg",
    },
}

DEFAULT_YOLO_SEG_MODEL = "yolo11"

YOLO_SEG_PRESET = {
    "size": "m",            # n/s/m/l/x — m balances mask quality vs. T4/P100 memory
    "epochs": 80,           # matches the YOLOX/Mask R-CNN detector presets
    "batch": 8,
    "imgsz": 1024,          # rect training keeps the 1:2 (512x1024) aspect ratio
    "rect": True,
    "learning_rate": 0.001,
    "input_height": 512,
    "input_width": 1024,
    # Inference postprocessing used during fixed-test evaluation.
    "conf_threshold": 0.25,
    "iou_threshold": 0.7,
    "min_polygon_area": 16,  # drop speckle contours when converting masks->labels
    "patience": 20,          # YOLO early-stopping (epochs without val improvement)
}


def get_segmentation_preset(model_name):
    if model_name not in SEGMENTATION_PRESETS:
        raise KeyError(f"Unknown segmentation model preset: {model_name}")
    return deepcopy(SEGMENTATION_PRESETS[model_name])


def get_yolo_seg_preset():
    return deepcopy(YOLO_SEG_PRESET)


def resolve_yolo_seg_model(model_name, size=None):
    """Map a friendly alias (yolo11/yolo26) to its ultralytics weights string.

    Also accepts a raw ultralytics spec (e.g. 'yolo11m-seg.pt' or a path),
    which is returned unchanged with a derived run name.
    """
    if model_name in YOLO_SEG_MODELS:
        spec = YOLO_SEG_MODELS[model_name]
        size = size or YOLO_SEG_PRESET["size"]
        return {
            "alias": model_name,
            "weights": spec["weights_template"].format(size=size),
            "run_name": spec["run_name"],
        }
    # Raw weights string / checkpoint path passthrough.
    stem = model_name.replace(".pt", "").replace("/", "_")
    return {"alias": model_name, "weights": model_name, "run_name": stem}


def get_mask_rcnn_preset():
    return deepcopy(MASK_RCNN_PRESET)


def get_yolox_preset():
    return deepcopy(YOLOX_PRESET)
