from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Multiply
from keras.models import Model


_ENCODER_FILTERS = (64, 128, 256, 512)
_BOTTLENECK_FILTERS = 512
_BB_CONV_DROPOUT = 0.4
_BB_CONV_BN_MOMENTUM = 0.1


def _conv_block(x, filters, name):
    x = Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = BatchNormalization(name=f"{name}_bn1")(x)
    x = Activation("relu", name=f"{name}_relu1")(x)
    x = Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = BatchNormalization(name=f"{name}_bn2")(x)
    x = Activation("relu", name=f"{name}_relu2")(x)
    return x


def _bb_conv(x, filters, pool_steps, name):
    for i in range(pool_steps):
        x = MaxPooling2D(pool_size=(2, 2), name=f"{name}_pool{i + 1}")(x)
    x = Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = BatchNormalization(momentum=_BB_CONV_BN_MOMENTUM, name=f"{name}_bn1")(x)
    x = Activation("relu", name=f"{name}_relu1")(x)
    x = Dropout(_BB_CONV_DROPOUT, name=f"{name}_drop1")(x)
    x = Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = BatchNormalization(momentum=_BB_CONV_BN_MOMENTUM, name=f"{name}_bn2")(x)
    x = Activation("relu", name=f"{name}_relu2")(x)
    x = Dropout(_BB_CONV_DROPOUT, name=f"{name}_drop2")(x)
    return x


def _decoder_block(x, skip, filters, name):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same", name=f"{name}_up")(x)
    x = Concatenate(name=f"{name}_concat")([x, skip])
    x = _conv_block(x, filters, name=f"{name}_conv")
    return x


def ICPRUnet(
    input_shape=(None, None, 3),
    classes=1,
    activation="sigmoid",
):
    inputs = Input(shape=input_shape, name="data")
    x = inputs
    skips = []

    for idx, filters in enumerate(_ENCODER_FILTERS, start=1):
        x = _conv_block(x, filters, name=f"enc{idx}")
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2), name=f"pool{idx}")(x)

    x = _conv_block(x, _BOTTLENECK_FILTERS, name="bottleneck")

    for idx in range(len(_ENCODER_FILTERS), 0, -1):
        x = _decoder_block(x, skips[idx - 1], _ENCODER_FILTERS[idx - 1], name=f"dec{idx}")

    x = Conv2D(classes, (1, 1), padding="same", name="final_conv")(x)
    x = Activation(activation, name=activation)(x)
    model = Model(inputs, x, name="icpr-unet")
    return model


def ICPRModifiedUnet(
    input_shape=(None, None, 3),
    classes=1,
    activation="sigmoid",
    bb_channels=None,
):
    bb_channels = bb_channels or classes
    inputs = Input(shape=input_shape, name="data")
    bb_input = Input(shape=input_shape[:2] + (bb_channels,), name="bb_input")
    x = inputs
    skips = []
    bb_feats = []

    for idx, filters in enumerate(_ENCODER_FILTERS, start=1):
        x = _conv_block(x, filters, name=f"enc{idx}")
        skips.append(x)
        bb_feats.append(_bb_conv(bb_input, filters, pool_steps=idx - 1, name=f"bbconv{idx}"))
        x = MaxPooling2D(pool_size=(2, 2), name=f"pool{idx}")(x)

    x = _conv_block(x, _BOTTLENECK_FILTERS, name="bottleneck")

    for idx in range(len(_ENCODER_FILTERS), 0, -1):
        skip = Multiply(name=f"bbconv_gate{idx}")([skips[idx - 1], bb_feats[idx - 1]])
        x = _decoder_block(x, skip, _ENCODER_FILTERS[idx - 1], name=f"dec{idx}")

    x = Conv2D(classes, (1, 1), padding="same", name="final_conv")(x)
    x = Activation(activation, name=activation)(x)
    model = Model([inputs, bb_input], x, name="icpr-munet")
    return model
