from keras import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Add
from keras.layers import Lambda
from keras.layers import Concatenate

try:
    from keras.utils import get_source_inputs
except ImportError:  # pragma: no cover
    def get_source_inputs(tensor):
        return tensor


def _obtain_input_shape(
    input_shape,
    default_size=224,
    min_size=197,
    data_format="channels_last",
    require_flatten=False,
):
    if input_shape is None:
        if data_format == "channels_first":
            input_shape = (3, default_size, default_size)
        else:
            input_shape = (default_size, default_size, 3)

    if len(input_shape) != 3:
        raise ValueError("Input shape must be a tuple of (height, width, channels).")

    if data_format == "channels_first":
        if input_shape[0] not in (1, 3) and require_flatten:
            raise ValueError("Input channels must be 1 or 3.")
        if input_shape[1] is not None and input_shape[1] < min_size:
            raise ValueError(f"Input height must be at least {min_size}.")
        if input_shape[2] is not None and input_shape[2] < min_size:
            raise ValueError(f"Input width must be at least {min_size}.")
        if require_flatten and (input_shape[1] is None or input_shape[2] is None):
            raise ValueError("When include_top=True, input spatial dimensions must be known.")
    else:
        if input_shape[-1] not in (1, 3) and require_flatten:
            raise ValueError("Input channels must be 1 or 3.")
        if input_shape[0] is not None and input_shape[0] < min_size:
            raise ValueError(f"Input height must be at least {min_size}.")
        if input_shape[1] is not None and input_shape[1] < min_size:
            raise ValueError(f"Input width must be at least {min_size}.")
        if require_flatten and (input_shape[0] is None or input_shape[1] is None):
            raise ValueError("When include_top=True, input spatial dimensions must be known.")

    return input_shape


def get_conv_params(**params):
    default_conv_params = {
        "kernel_initializer": "glorot_uniform",
        "use_bias": False,
        "padding": "valid",
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        "axis": 3,
        "momentum": 0.99,
        "epsilon": 2e-5,
        "center": True,
        "scale": True,
    }
    default_bn_params.update(params)
    return default_bn_params


def handle_block_names(stage, block):
    name_base = f"stage{stage + 1}_unit{block + 1}_"
    conv_name = name_base + "conv"
    bn_name = name_base + "bn"
    relu_name = name_base + "relu"
    sc_name = name_base + "sc"
    return conv_name, bn_name, relu_name, sc_name


def GroupConv2D(filters, kernel_size, conv_params, conv_name, strides=(1, 1), cardinality=32):
    def layer(input_tensor):
        input_channels = int(input_tensor.shape[-1])
        grouped_channels = input_channels // cardinality
        if grouped_channels < 1:
            raise ValueError("Cardinality is too large for the input channels.")

        blocks = []
        for c in range(cardinality):
            x = Lambda(
                lambda z, c=c: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
            )(input_tensor)
            name = conv_name + "_" + str(c)
            x = Conv2D(
                grouped_channels,
                kernel_size,
                strides=strides,
                name=name,
                **conv_params,
            )(x)
            blocks.append(x)

        x = Concatenate(axis=-1)(blocks)
        return x

    return layer


def conv_block(filters, stage, block, strides=(2, 2)):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = Conv2D(filters, (1, 1), name=conv_name + "1", **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + "1", **bn_params)(x)
        x = Activation("relu", name=relu_name + "1")(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(
            filters,
            (3, 3),
            conv_params,
            conv_name + "2",
            strides=strides,
        )(x)
        x = BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = Activation("relu", name=relu_name + "2")(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + "3", **conv_params)(x)
        x = BatchNormalization(name=bn_name + "3", **bn_params)(x)

        shortcut = Conv2D(filters * 2, (1, 1), name=sc_name, strides=strides, **conv_params)(input_tensor)
        shortcut = BatchNormalization(name=sc_name + "_bn", **bn_params)(shortcut)
        x = Add()([x, shortcut])

        x = Activation("relu", name=relu_name)(x)
        return x

    return layer


def identity_block(filters, stage, block):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, _ = handle_block_names(stage, block)

        x = Conv2D(filters, (1, 1), name=conv_name + "1", **conv_params)(input_tensor)
        x = BatchNormalization(name=bn_name + "1", **bn_params)(x)
        x = Activation("relu", name=relu_name + "1")(x)

        x = ZeroPadding2D(padding=(1, 1))(x)
        x = GroupConv2D(filters, (3, 3), conv_params, conv_name + "2")(x)
        x = BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = Activation("relu", name=relu_name + "2")(x)

        x = Conv2D(filters * 2, (1, 1), name=conv_name + "3", **conv_params)(x)
        x = BatchNormalization(name=bn_name + "3", **bn_params)(x)

        x = Add()([x, input_tensor])
        x = Activation("relu", name=relu_name)(x)
        return x

    return layer


def build_resnext(
    repetitions=(2, 2, 2, 2),
    include_top=True,
    input_tensor=None,
    input_shape=None,
    classes=1000,
    first_conv_filters=64,
    first_block_filters=64,
):
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=197,
        data_format="channels_last",
        require_flatten=include_top,
    )

    if input_tensor is None:
        img_input = Input(shape=input_shape, name="data")
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    no_scale_bn_params = get_bn_params(scale=False)
    bn_params = get_bn_params()
    conv_params = get_conv_params()
    init_filters = first_block_filters

    x = BatchNormalization(name="bn_data", **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(first_conv_filters, (7, 7), strides=(2, 2), name="conv0", **conv_params)(x)
    x = BatchNormalization(name="bn0", **bn_params)(x)
    x = Activation("relu", name="relu0")(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="pooling0")(x)

    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = init_filters * (2**stage)
            if stage == 0 and block == 0:
                x = conv_block(filters, stage, block, strides=(1, 1))(x)
            elif block == 0:
                x = conv_block(filters, stage, block, strides=(2, 2))(x)
            else:
                x = identity_block(filters, stage, block)(x)

    if include_top:
        x = GlobalAveragePooling2D(name="pool1")(x)
        x = Dense(classes, name="fc1")(x)
        x = Activation("softmax", name="softmax")(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x)
    return model


def ResNeXt50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights is not None:
        raise NotImplementedError("Weights loading is not implemented for ResNeXt50.")
    model = build_resnext(
        input_tensor=input_tensor,
        input_shape=input_shape,
        first_block_filters=128,
        repetitions=(3, 4, 6, 3),
        classes=classes,
        include_top=include_top,
    )
    model.name = "resnext50"
    return model


def ResNeXt101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights is not None:
        raise NotImplementedError("Weights loading is not implemented for ResNeXt101.")
    model = build_resnext(
        input_tensor=input_tensor,
        input_shape=input_shape,
        first_block_filters=128,
        repetitions=(3, 4, 23, 3),
        classes=classes,
        include_top=include_top,
    )
    model.name = "resnext101"
    return model
