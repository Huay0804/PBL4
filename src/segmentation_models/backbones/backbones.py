import importlib
import re


_LOCAL_CLASSIFICATION_MODULE = (
    f"{__package__}.classification_models"
    if __package__
    else "segmentation_models.backbones.classification_models"
)

_MODEL_SPECS = {
    "vgg16": ("keras.applications", "VGG16"),
    "vgg19": ("keras.applications", "VGG19"),
    "resnet18": (_LOCAL_CLASSIFICATION_MODULE, "ResNet18"),
    "resnet34": (_LOCAL_CLASSIFICATION_MODULE, "ResNet34"),
    "resnet50": (_LOCAL_CLASSIFICATION_MODULE, "ResNet50"),
    "resnet101": (_LOCAL_CLASSIFICATION_MODULE, "ResNet101"),
    "resnet152": (_LOCAL_CLASSIFICATION_MODULE, "ResNet152"),
    "resnext50": (_LOCAL_CLASSIFICATION_MODULE, "ResNeXt50"),
    "resnext101": (_LOCAL_CLASSIFICATION_MODULE, "ResNeXt101"),
    "inceptionresnetv2": ("keras.applications", "InceptionResNetV2"),
    "inceptionv3": ("keras.applications", "InceptionV3"),
    "densenet121": ("keras.applications", "DenseNet121"),
    "densenet169": ("keras.applications", "DenseNet169"),
    "densenet201": ("keras.applications", "DenseNet201"),
}


def _normalize_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Backbone name must be a non-empty string.")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _load_class(module_path, class_name, backbone_name=None):
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if module_path == _LOCAL_CLASSIFICATION_MODULE:
            name = backbone_name or class_name
            raise NotImplementedError(
                f"Backbone '{name}' is listed for compatibility but not implemented yet. "
                "Implement it under segmentation_models/backbones/classification_models."
            ) from exc
        raise
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        if module_path == _LOCAL_CLASSIFICATION_MODULE:
            name = backbone_name or class_name
            raise NotImplementedError(
                f"Backbone '{name}' is listed for compatibility but not implemented yet. "
                "Implement it under segmentation_models/backbones/classification_models."
            ) from exc
        raise ValueError(
            f"Backbone class {class_name} not found in {module_path}. "
            "Check your Keras version."
        ) from exc


def list_backbones():
    return sorted(_MODEL_SPECS.keys())


def is_backbone_supported(name):
    key = _normalize_name(name)
    spec = _MODEL_SPECS.get(key)
    if spec is None:
        return False
    module_path, class_name = spec
    try:
        _load_class(module_path, class_name, backbone_name=key)
    except (NotImplementedError, ModuleNotFoundError, ValueError):
        return False
    return True


def get_backbone(name, *args, **kwargs):
    key = _normalize_name(name)
    spec = _MODEL_SPECS.get(key)
    if spec is None:
        supported = ", ".join(list_backbones())
        raise ValueError(f"Unknown backbone '{name}'. Supported: {supported}")
    module_path, class_name = spec
    backbone_cls = _load_class(module_path, class_name, backbone_name=key)
    return backbone_cls(*args, **kwargs)
