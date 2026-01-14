import numpy as np
from keras.layers import Conv2D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import SpatialDropout2D
from keras.models import Model

from .blocks import pyramid_block
from ..common import ResizeImage
from ..common import Conv2DBlock
from ..utils import extract_outputs
from ..utils import to_tuple


def build_fpn(
    backbone,
    fpn_layers,
    classes=21,
    activation="softmax",
    upsample_rates=(2, 2, 2),
    last_upsample=4,
    pyramid_filters=256,
    segmentation_filters=128,
    use_batchnorm=False,
    dropout=None,
    interpolation="bilinear",
):
    """
    Implementation of FPN head for segmentation models according to:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
    """
    if len(upsample_rates) != len(fpn_layers):
        raise ValueError("Number of intermediate feature maps and upsample steps should match")

    outputs = extract_outputs(backbone, fpn_layers, include_top=True)

    upsample_rates = [1] + list(upsample_rates)

    m = None
    pyramid = []
    for i, c in enumerate(outputs):
        m, p = pyramid_block(
            pyramid_filters=pyramid_filters,
            segmentation_filters=segmentation_filters,
            upsample_rate=upsample_rates[i],
            use_batchnorm=use_batchnorm,
            stage=i,
        )(c, m)
        pyramid.append(p)

    upsampled_pyramid = []
    for i, p in enumerate(pyramid[::-1]):
        if upsample_rates[i] > 1:
            upsample_rate = to_tuple(np.prod(upsample_rates[: i + 1]))
            p = ResizeImage(upsample_rate, interpolation=interpolation)(p)
        upsampled_pyramid.append(p)

    x = Concatenate()(upsampled_pyramid)

    n_filters = segmentation_filters * len(pyramid)
    x = Conv2DBlock(n_filters, (3, 3), use_batchnorm=use_batchnorm, padding="same")(x)
    if dropout is not None:
        x = SpatialDropout2D(dropout)(x)

    x = Conv2D(classes, (3, 3), padding="same")(x)

    x = ResizeImage(to_tuple(last_upsample), interpolation=interpolation)(x)

    x = Activation(activation)(x)

    model = Model(backbone.input, x)
    return model
