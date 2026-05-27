from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Multiply
from keras.models import Model

from .blocks import ConvRelu
from .blocks import Transpose2D_block
from .blocks import Upsample2D_block
from ..unet.blocks import BBConv
from ..utils import to_tuple


def _encoder_block(x, filters, stage, use_batchnorm):
    x = ConvRelu(
        filters,
        (3, 3),
        use_batchnorm=use_batchnorm,
        conv_name=f"encoder_stage{stage}_conv1",
        bn_name=f"encoder_stage{stage}_bn1",
        relu_name=f"encoder_stage{stage}_relu1",
    )(x)
    x = ConvRelu(
        filters,
        (3, 3),
        use_batchnorm=use_batchnorm,
        conv_name=f"encoder_stage{stage}_conv2",
        bn_name=f"encoder_stage{stage}_bn2",
        relu_name=f"encoder_stage{stage}_relu2",
    )(x)
    return x


def _make_supervision_outputs(supervision_nodes, classes, activation):
    outputs = []
    for idx, node in enumerate(supervision_nodes, start=1):
        is_final_head = idx == len(supervision_nodes)
        conv_name = "final_conv" if is_final_head else f"aux_conv_ds{idx}"
        act_name = "final_output" if is_final_head else f"aux_output_ds{idx}"
        head = Conv2D(classes, (1, 1), padding="same", name=conv_name)(node)
        head = Activation(activation, name=act_name)(head)
        outputs.append(head)
    return outputs


def build_nestnet(
    classes,
    input_tensor=None,
    encoder_filters=(64, 128, 256, 512, 512),
    decoder_filters=(512, 256, 128, 64),
    upsample_rates=(2, 2, 2, 2),
    n_upsample_blocks=4,
    block_type="upsampling",
    activation="sigmoid",
    use_batchnorm=True,
    bb_input=None,
    bb_use_batchnorm=True,
    bb_kernel_size=(3, 3),
    deep_supervision=False,
    deep_supervision_branches=4,
    deep_supervision_output_index=None,
):
    if input_tensor is None:
        raise ValueError("input_tensor is required.")
    if n_upsample_blocks != 4:
        raise ValueError("ICPR-aligned NestNet uses exactly 4 upsampling blocks.")
    if len(encoder_filters) != 5:
        raise ValueError("encoder_filters must define 5 levels.")
    if len(decoder_filters) != 4:
        raise ValueError("decoder_filters must define 4 levels.")
    if len(upsample_rates) != 4:
        raise ValueError("upsample_rates must define 4 levels.")

    if block_type == "transpose":
        up_block = Transpose2D_block
    elif block_type == "upsampling":
        up_block = Upsample2D_block
    else:
        raise ValueError(f"Unsupported decoder block type: {block_type}")

    nodes = [[None for _ in range(5)] for _ in range(5)]

    # Encoder column Xi,0 with fixed ICPR-style channel progression.
    nodes[0][0] = _encoder_block(input_tensor, encoder_filters[0], 0, use_batchnorm)
    for row in range(1, 5):
        pooled = MaxPooling2D(pool_size=(2, 2), name=f"encoder_stage{row}_pool")(nodes[row - 1][0])
        nodes[row][0] = _encoder_block(pooled, encoder_filters[row], row, use_batchnorm)

    if bb_input is not None:
        for row in range(4):
            bb_features = BBConv(
                encoder_filters[row],
                pool_steps=row,
                kernel_size=bb_kernel_size,
                use_batchnorm=bb_use_batchnorm,
                name=f"bbconv_stage{row}",
            )(bb_input)
            nodes[row][0] = Multiply(name=f"bbconv_gate_stage{row}")([nodes[row][0], bb_features])

    max_supervision_col = min(4, deep_supervision_branches)
    if max_supervision_col < 1:
        raise ValueError("deep_supervision_branches must be at least 1.")
    if deep_supervision and deep_supervision_output_index is not None:
        raise ValueError("deep_supervision_output_index can only be used with a single output.")
    if deep_supervision_output_index is not None:
        if deep_supervision_output_index < 0 or deep_supervision_output_index >= max_supervision_col:
            raise ValueError(
                "deep_supervision_output_index must be between 0 and "
                f"{max_supervision_col - 1}, got {deep_supervision_output_index}."
            )
        build_supervision_col = deep_supervision_output_index + 1
    else:
        build_supervision_col = max_supervision_col

    for col in range(1, build_supervision_col + 1):
        for row in range(4 - col + 1):
            below = nodes[row + 1][col - 1]
            dense_skips = [nodes[row][prev_col] for prev_col in range(col) if nodes[row][prev_col] is not None]
            decoder_idx = 3 - row
            nodes[row][col] = up_block(
                decoder_filters[decoder_idx],
                decoder_idx + 1,
                col,
                upsample_rate=to_tuple(upsample_rates[decoder_idx]),
                skip=dense_skips or None,
                use_batchnorm=use_batchnorm,
            )(below)

    supervision_nodes = [
        nodes[0][col] for col in range(1, build_supervision_col + 1) if nodes[0][col] is not None
    ]
    if len(supervision_nodes) != build_supervision_col:
        raise ValueError("Failed to build top-row deep-supervision nodes.")

    outputs = _make_supervision_outputs(supervision_nodes, classes, activation)
    inputs = [input_tensor]
    if bb_input is not None:
        inputs.append(bb_input)
    if deep_supervision:
        model_outputs = outputs
    elif deep_supervision_output_index is not None:
        model_outputs = outputs[deep_supervision_output_index]
    else:
        model_outputs = outputs[-1]
    return Model(inputs, model_outputs)
