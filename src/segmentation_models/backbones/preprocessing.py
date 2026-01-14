import importlib
import re


def _identical(x):
    return x


def _bgr_transpose(x):
    return x[..., ::-1]


_CUSTOM_PREPROCESSING = {
    "resnet18": _bgr_transpose,
    "resnet34": _bgr_transpose,
    "resnet50": _bgr_transpose,
    "resnet101": _bgr_transpose,
    "resnet152": _bgr_transpose,
    "resnext50": _identical,
    "resnext101": _identical,
}


_PREPROCESSING_SPECS = {
    "vgg16": "keras.applications.vgg16",
    "vgg19": "keras.applications.vgg19",
    "densenet121": "keras.applications.densenet",
    "densenet169": "keras.applications.densenet",
    "densenet201": "keras.applications.densenet",
    "inceptionv3": "keras.applications.inception_v3",
    "inceptionresnetv2": "keras.applications.inception_resnet_v2",
}


def _normalize_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Backbone name must be a non-empty string.")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def get_preprocessing(name):
    key = _normalize_name(name)
    custom = _CUSTOM_PREPROCESSING.get(key)
    if custom is not None:
        return custom
    module_path = _PREPROCESSING_SPECS.get(key)
    if module_path is None:
        supported = ", ".join(sorted(set(_CUSTOM_PREPROCESSING) | set(_PREPROCESSING_SPECS)))
        raise ValueError(f"Unknown backbone '{name}'. Supported: {supported}")

    module = importlib.import_module(module_path)
    if not hasattr(module, "preprocess_input"):
        raise ValueError(
            f"Backbone '{name}' does not expose a preprocess_input in {module_path}."
        )
    return module.preprocess_input
