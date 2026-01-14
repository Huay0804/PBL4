import math

from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import Multiply
from keras.models import Model

from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from .blocks import BBConv
from ..utils import get_layer_number
from ..utils import to_tuple


def _infer_bb_pool_steps(bb_input, skip):
    bb_height = bb_input.shape[1]
    bb_width = bb_input.shape[2]
    skip_height = skip.shape[1]
    skip_width = skip.shape[2]
    if None in (bb_height, bb_width, skip_height, skip_width):
        return None
    if bb_height % skip_height != 0 or bb_width % skip_width != 0:
        return None
    ratio_h = bb_height // skip_height
    ratio_w = bb_width // skip_width
    if ratio_h != ratio_w or ratio_h <= 0:
        return None
    steps = int(round(math.log2(ratio_h)))
    if 2**steps != ratio_h:
        return None
    return steps


def build_unet(
    backbone,
    classes,
    skip_connection_layers,
    decoder_filters=(256, 128, 64, 32, 16),
    upsample_rates=(2, 2, 2, 2, 2),
    n_upsample_blocks=5,
    block_type="upsampling",
    activation="sigmoid",
    use_batchnorm=True,
    bb_input=None,
    bb_use_batchnorm=True,
    bb_kernel_size=(3, 3),
):
    input = backbone.input
    x = backbone.output

    if block_type == "transpose":
        up_block = Transpose2D_block
    else:
        up_block = Upsample2D_block

    skip_connection_idx = [
        get_layer_number(backbone, layer) if isinstance(layer, str) else layer
        for layer in skip_connection_layers
    ]

    for i in range(n_upsample_blocks):
        skip_connection = None
        if i < len(skip_connection_idx):
            skip_connection = backbone.layers[skip_connection_idx[i]].output

        if skip_connection is not None and bb_input is not None:
            pool_steps = _infer_bb_pool_steps(bb_input, skip_connection)
            if pool_steps is None:
                raise ValueError("BB-Conv requires power-of-two downsampling between bb_input and skip.")
            filters = skip_connection.shape[-1]
            if filters is None:
                raise ValueError("BB-Conv requires a known channel count for skip connections.")
            bb_features = BBConv(
                int(filters),
                pool_steps=pool_steps,
                kernel_size=bb_kernel_size,
                use_batchnorm=bb_use_batchnorm,
                name=f"bbconv_stage{i}",
            )(bb_input)
            skip_connection = Multiply(name=f"bbconv_gate_stage{i}")([skip_connection, bb_features])

        upsample_rate = to_tuple(upsample_rates[i])

        x = up_block(
            decoder_filters[i],
            i,
            upsample_rate=upsample_rate,
            skip=skip_connection,
            use_batchnorm=use_batchnorm,
        )(x)

    x = Conv2D(classes, (3, 3), padding="same", name="final_conv")(x)
    x = Activation(activation, name=activation)(x)

    inputs = [input]
    if bb_input is not None:
        inputs.append(bb_input)
    model = Model(inputs, x)
    return model
