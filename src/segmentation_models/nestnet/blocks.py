from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate


def handle_block_names(stage, cols):
    conv_name = f"decoder_stage{stage}-{cols}_conv"
    bn_name = f"decoder_stage{stage}-{cols}_bn"
    relu_name = f"decoder_stage{stage}-{cols}_relu"
    up_name = f"decoder_stage{stage}-{cols}_upsample"
    merge_name = f"merge_{stage}-{cols}"
    return conv_name, bn_name, relu_name, up_name, merge_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name="conv", bn_name="bn", relu_name="relu"):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not use_batchnorm)(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation("relu", name=relu_name)(x)
        return x

    return layer


def _as_skip_list(skip):
    if skip is None:
        return []
    if isinstance(skip, (list, tuple)):
        return [tensor for tensor in skip if tensor is not None]
    return [skip]


def Upsample2D_block(filters, stage, cols, kernel_size=(3, 3), upsample_rate=(2, 2), use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        skip_tensors = _as_skip_list(skip)
        if skip_tensors:
            x = Concatenate(name=merge_name)([x] + skip_tensors)

        x = ConvRelu(
            filters,
            kernel_size,
            use_batchnorm=use_batchnorm,
            conv_name=conv_name + "1",
            bn_name=bn_name + "1",
            relu_name=relu_name + "1",
        )(x)

        x = ConvRelu(
            filters,
            kernel_size,
            use_batchnorm=use_batchnorm,
            conv_name=conv_name + "2",
            bn_name=bn_name + "2",
            relu_name=relu_name + "2",
        )(x)

        return x

    return layer


def Transpose2D_block(
    filters,
    stage,
    cols,
    kernel_size=(3, 3),
    upsample_rate=(2, 2),
    transpose_kernel_size=(4, 4),
    use_batchnorm=False,
    skip=None,
):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name, merge_name = handle_block_names(stage, cols)

        x = Conv2DTranspose(
            filters,
            transpose_kernel_size,
            strides=upsample_rate,
            padding="same",
            name=up_name,
            use_bias=not use_batchnorm,
        )(input_tensor)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name + "1")(x)
        x = Activation("relu", name=relu_name + "1")(x)

        skip_tensors = _as_skip_list(skip)
        if skip_tensors:
            x = Concatenate(name=merge_name)([x] + skip_tensors)

        x = ConvRelu(
            filters,
            kernel_size,
            use_batchnorm=use_batchnorm,
            conv_name=conv_name + "2",
            bn_name=bn_name + "2",
            relu_name=relu_name + "2",
        )(x)

        return x

    return layer
