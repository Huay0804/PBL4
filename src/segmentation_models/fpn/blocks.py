from keras.layers import Add

from ..common import Conv2DBlock
from ..common import ResizeImage
from ..utils import to_tuple


def pyramid_block(pyramid_filters=256, segmentation_filters=128, upsample_rate=2, use_batchnorm=False, stage=0):
    """
    Pyramid block according to:
        http://presentations.cocodataset.org/COCO17-Stuff-FAIR.pdf
    """
    def layer(c, m=None):
        x = Conv2DBlock(
            pyramid_filters,
            (1, 1),
            padding="same",
            use_batchnorm=use_batchnorm,
            name="pyramid_stage_{}".format(stage),
        )(c)

        if m is not None:
            up = ResizeImage(to_tuple(upsample_rate))(m)
            x = Add()([x, up])

        p = Conv2DBlock(
            segmentation_filters,
            (3, 3),
            padding="same",
            use_batchnorm=use_batchnorm,
            name="segm1_stage_{}".format(stage),
        )(x)

        p = Conv2DBlock(
            segmentation_filters,
            (3, 3),
            padding="same",
            use_batchnorm=use_batchnorm,
            name="segm2_stage_{}".format(stage),
        )(p)
        m = x

        return m, p

    return layer
