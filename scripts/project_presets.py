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


def get_segmentation_preset(model_name):
    if model_name not in SEGMENTATION_PRESETS:
        raise KeyError(f"Unknown segmentation model preset: {model_name}")
    return deepcopy(SEGMENTATION_PRESETS[model_name])


def get_mask_rcnn_preset():
    return deepcopy(MASK_RCNN_PRESET)


def get_yolox_preset():
    return deepcopy(YOLOX_PRESET)
