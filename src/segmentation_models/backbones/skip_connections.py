UNET_SKIP_CONNECTIONS = {
    "vgg16": ("block5_conv3", "block4_conv3", "block3_conv3", "block2_conv2", "block1_conv2"),
    "vgg19": ("block5_conv4", "block4_conv4", "block3_conv4", "block2_conv2", "block1_conv2"),
    "resnet18": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet34": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet50": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet101": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet152": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnext50": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnext101": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "inceptionv3": (228, 86, 16, 9),
    "inceptionresnetv2": (594, 260, 16, 9),
    "densenet121": (311, 139, 51, 4),
    "densenet169": (367, 139, 51, 4),
    "densenet201": (479, 139, 51, 4),
}

LINKNET_SKIP_CONNECTIONS = {
    "vgg16": ("block5_conv3", "block4_conv3", "block3_conv3", "block2_conv2"),
    "vgg19": ("block5_conv4", "block4_conv4", "block3_conv4", "block2_conv2"),
    "resnet18": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet34": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet50": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet101": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnet152": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnext50": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "resnext101": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1", "relu0"),
    "inceptionv3": (228, 86, 16, 9),
    "inceptionresnetv2": (594, 260, 16, 9),
    "densenet121": (311, 139, 51, 4),
    "densenet169": (367, 139, 51, 4),
    "densenet201": (479, 139, 51, 4),
}

NESTNET_SKIP_CONNECTIONS = {
    "vgg16": (
        "block5_conv3",
        "block4_conv3",
        "block3_conv3",
        "block2_conv2",
        "block1_conv2",
        "block5_pool",
        "block4_pool",
        "block3_pool",
        "block2_pool",
        "block1_pool",
    ),
    "vgg19": (
        "block5_conv4",
        "block4_conv4",
        "block3_conv4",
        "block2_conv2",
        "block1_conv2",
        "block5_pool",
        "block4_pool",
        "block3_pool",
        "block2_pool",
        "block1_pool",
    ),
    "resnet18": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet34": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet50": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet101": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet152": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnext50": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "stage4_unit1_relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnext101": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "stage4_unit1_relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "inceptionv3": (228, 86, 16, 9),
    "inceptionresnetv2": (594, 260, 16, 9),
    "densenet121": (311, 139, 51, 4),
    "densenet169": (367, 139, 51, 4),
    "densenet201": (479, 139, 51, 4),
}

XNET_SKIP_CONNECTIONS = {
    "vgg16": (
        "block5_conv3",
        "block4_conv3",
        "block3_conv3",
        "block2_conv2",
        "block1_conv2",
        "block5_pool",
        "block4_pool",
        "block3_pool",
        "block2_pool",
        "block1_pool",
    ),
    "vgg19": (
        "block5_conv4",
        "block4_conv4",
        "block3_conv4",
        "block2_conv2",
        "block1_conv2",
        "block5_pool",
        "block4_pool",
        "block3_pool",
        "block2_pool",
        "block1_pool",
    ),
    "resnet18": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet34": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet50": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet101": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnet152": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnext50": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "stage4_unit1_relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "resnext101": (
        "stage4_unit1_relu1",
        "stage3_unit1_relu1",
        "stage2_unit1_relu1",
        "relu0",
        "stage4_unit1_relu1",
        "stage3_unit2_relu1",
        "stage2_unit2_relu1",
        "stage1_unit2_relu1",
    ),
    "inceptionv3": (228, 86, 16, 9),
    "inceptionresnetv2": (594, 260, 16, 9),
    "densenet121": (311, 139, 51, 4),
    "densenet169": (367, 139, 51, 4),
    "densenet201": (479, 139, 51, 4),
}


_ARCH_SKIP_CONNECTIONS = {
    "unet": UNET_SKIP_CONNECTIONS,
    "linknet": LINKNET_SKIP_CONNECTIONS,
    "nestnet": NESTNET_SKIP_CONNECTIONS,
    "xnet": XNET_SKIP_CONNECTIONS,
}


def list_supported_backbones(architecture="unet"):
    if architecture not in _ARCH_SKIP_CONNECTIONS:
        raise ValueError(f"Unknown architecture '{architecture}'.")
    return sorted(_ARCH_SKIP_CONNECTIONS[architecture].keys())


def get_skip_connections(backbone_name, architecture="unet"):
    if architecture not in _ARCH_SKIP_CONNECTIONS:
        raise ValueError(f"Unknown architecture '{architecture}'.")
    mapping = _ARCH_SKIP_CONNECTIONS[architecture]
    try:
        return mapping[backbone_name]
    except KeyError as exc:
        supported = ", ".join(sorted(mapping))
        raise ValueError(
            f"Unknown backbone '{backbone_name}' for {architecture}. Supported: {supported}"
        ) from exc
