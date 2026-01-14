from .builder import build_fpn
from ..backbones import get_backbone
from ..utils import freeze_model


DEFAULT_FEATURE_PYRAMID_LAYERS = {
    "vgg16": ("block5_conv3", "block4_conv3", "block3_conv3"),
    "vgg19": ("block5_conv4", "block4_conv4", "block3_conv4"),
    "resnet18": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1"),
    "resnet34": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1"),
    "resnet50": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1"),
    "resnet101": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1"),
    "resnet152": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1"),
    "resnext50": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1"),
    "resnext101": ("stage4_unit1_relu1", "stage3_unit1_relu1", "stage2_unit1_relu1"),
    "inceptionv3": (228, 86, 16),
    "inceptionresnetv2": (594, 260, 16),
    "densenet121": (311, 139, 51),
    "densenet169": (367, 139, 51),
    "densenet201": (479, 139, 51),
}


def FPN(
    backbone_name="vgg16",
    input_shape=(None, None, 3),
    input_tensor=None,
    encoder_weights="imagenet",
    freeze_encoder=False,
    fpn_layers="default",
    pyramid_block_filters=256,
    segmentation_block_filters=128,
    upsample_rates=(2, 2, 2),
    last_upsample=4,
    interpolation="bilinear",
    use_batchnorm=True,
    classes=21,
    activation="softmax",
    dropout=None,
):
    """
    Implementation of FPN head for segmentation models.
    """
    backbone = get_backbone(
        backbone_name,
        input_shape=input_shape,
        input_tensor=input_tensor,
        weights=encoder_weights,
        include_top=False,
    )

    if fpn_layers == "default":
        fpn_layers = DEFAULT_FEATURE_PYRAMID_LAYERS[backbone_name]

    model = build_fpn(
        backbone,
        fpn_layers,
        classes=classes,
        pyramid_filters=pyramid_block_filters,
        segmentation_filters=segmentation_block_filters,
        upsample_rates=upsample_rates,
        use_batchnorm=use_batchnorm,
        dropout=dropout,
        last_upsample=last_upsample,
        interpolation=interpolation,
        activation=activation,
    )

    if freeze_encoder:
        freeze_model(backbone)

    model.name = "fpn-{}".format(backbone.name)
    return model
