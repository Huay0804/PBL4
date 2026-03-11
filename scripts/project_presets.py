from copy import deepcopy


DEFAULT_SEGMENTATION_MODEL = "unet"

SEGMENTATION_PRESETS = {
    "unet": {
        "backbone": "resnet18",
        "batch_size": 2,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
    },
    "mod_unet": {
        "backbone": "resnet18",
        "batch_size": 2,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
    },
    "nestnet": {
        "backbone": "resnet18",
        "batch_size": 2,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
    },
    "linknet": {
        "backbone": "resnet18",
        "batch_size": 2,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
    },
    "fpn": {
        "backbone": "resnet18",
        "batch_size": 2,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
    },
    "icpr_unet": {
        "backbone": "resnet18",
        "batch_size": 2,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
    },
    "icpr_munet": {
        "backbone": "resnet18",
        "batch_size": 2,
        "epochs": 60,
        "learning_rate": 1e-4,
        "loss": "ce_dice",
    },
}

MASK_RCNN_PRESET = {
    "epochs": 80,
    "batch_size": 1,
    "learning_rate": 0.001,
}


def get_segmentation_preset(model_name):
    if model_name not in SEGMENTATION_PRESETS:
        raise KeyError(f"Unknown segmentation model preset: {model_name}")
    return deepcopy(SEGMENTATION_PRESETS[model_name])


def get_mask_rcnn_preset():
    return deepcopy(MASK_RCNN_PRESET)
