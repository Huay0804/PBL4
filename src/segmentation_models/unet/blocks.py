from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import MaxPooling2D


def handle_block_names(stage):
    conv_name = "decoder_stage{}_conv".format(stage)
    bn_name = "decoder_stage{}_bn".format(stage)
    relu_name = "decoder_stage{}_relu".format(stage)
    up_name = "decoder_stage{}_upsample".format(stage)
    return conv_name, bn_name, relu_name, up_name


def ConvRelu(filters, kernel_size, use_batchnorm=False, conv_name="conv", bn_name="bn", relu_name="relu"):
    def layer(x):
        x = Conv2D(filters, kernel_size, padding="same", name=conv_name, use_bias=not use_batchnorm)(x)
        if use_batchnorm:
            x = BatchNormalization(name=bn_name)(x)
        x = Activation("relu", name=relu_name)(x)
        return x

    return layer


def Upsample2D_block(filters, stage, kernel_size=(3, 3), upsample_rate=(2, 2), use_batchnorm=False, skip=None):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

        x = UpSampling2D(size=upsample_rate, name=up_name)(input_tensor)

        if skip is not None:
            x = Concatenate()([x, skip])

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
    kernel_size=(3, 3),
    upsample_rate=(2, 2),
    transpose_kernel_size=(4, 4),
    use_batchnorm=False,
    skip=None,
):
    def layer(input_tensor):
        conv_name, bn_name, relu_name, up_name = handle_block_names(stage)

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

        if skip is not None:
            x = Concatenate()([x, skip])

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


def BBConv(filters, pool_steps, kernel_size=(3, 3), use_batchnorm=False, name="bbconv"):
    def layer(x):
        for i in range(pool_steps):
            x = MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool{i + 1}")(x)

        x = Conv2D(
            filters,
            kernel_size,
            padding="same",
            name=f"{name}_conv1",
            use_bias=not use_batchnorm,
        )(x)
        if use_batchnorm:
            x = BatchNormalization(name=f"{name}_bn1")(x)
        x = Activation("relu", name=f"{name}_relu1")(x)

        x = Conv2D(
            filters,
            kernel_size,
            padding="same",
            name=f"{name}_conv2",
            use_bias=not use_batchnorm,
        )(x)
        if use_batchnorm:
            x = BatchNormalization(name=f"{name}_bn2")(x)
        x = Activation("relu", name=f"{name}_relu2")(x)
        return x

    return layer
