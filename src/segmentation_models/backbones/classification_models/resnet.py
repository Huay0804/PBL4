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


def basic_identity_block(filters, stage, block):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, _ = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + "1", **bn_params)(input_tensor)
        x = Activation("relu", name=relu_name + "1")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + "1", **conv_params)(x)

        x = BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = Activation("relu", name=relu_name + "2")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + "2", **conv_params)(x)

        x = Add()([x, input_tensor])
        return x

    return layer


def basic_conv_block(filters, stage, block, strides=(2, 2)):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + "1", **bn_params)(input_tensor)
        x = Activation("relu", name=relu_name + "1")(x)
        shortcut = x
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + "1", **conv_params)(x)

        x = BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = Activation("relu", name=relu_name + "2")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + "2", **conv_params)(x)

        shortcut = Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = Add()([x, shortcut])
        return x

    return layer


def conv_block(filters, stage, block, strides=(2, 2)):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + "1", **bn_params)(input_tensor)
        x = Activation("relu", name=relu_name + "1")(x)
        shortcut = x
        x = Conv2D(filters, (1, 1), name=conv_name + "1", **conv_params)(x)

        x = BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = Activation("relu", name=relu_name + "2")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), strides=strides, name=conv_name + "2", **conv_params)(x)

        x = BatchNormalization(name=bn_name + "3", **bn_params)(x)
        x = Activation("relu", name=relu_name + "3")(x)
        x = Conv2D(filters * 4, (1, 1), name=conv_name + "3", **conv_params)(x)

        shortcut = Conv2D(filters * 4, (1, 1), name=sc_name, strides=strides, **conv_params)(shortcut)
        x = Add()([x, shortcut])
        return x

    return layer


def identity_block(filters, stage, block):
    def layer(input_tensor):
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, _ = handle_block_names(stage, block)

        x = BatchNormalization(name=bn_name + "1", **bn_params)(input_tensor)
        x = Activation("relu", name=relu_name + "1")(x)
        x = Conv2D(filters, (1, 1), name=conv_name + "1", **conv_params)(x)

        x = BatchNormalization(name=bn_name + "2", **bn_params)(x)
        x = Activation("relu", name=relu_name + "2")(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Conv2D(filters, (3, 3), name=conv_name + "2", **conv_params)(x)

        x = BatchNormalization(name=bn_name + "3", **bn_params)(x)
        x = Activation("relu", name=relu_name + "3")(x)
        x = Conv2D(filters * 4, (1, 1), name=conv_name + "3", **conv_params)(x)

        x = Add()([x, input_tensor])
        return x

    return layer


def build_resnet(
    repetitions=(2, 2, 2, 2),
    include_top=True,
    input_tensor=None,
    input_shape=None,
    classes=1000,
    block_type="usual",
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
    init_filters = 64

    if block_type == "basic":
        conv_block_fn = basic_conv_block
        identity_block_fn = basic_identity_block
    else:
        conv_block_fn = conv_block
        identity_block_fn = identity_block

    x = BatchNormalization(name="bn_data", **no_scale_bn_params)(img_input)
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(init_filters, (7, 7), strides=(2, 2), name="conv0", **conv_params)(x)
    x = BatchNormalization(name="bn0", **bn_params)(x)
    x = Activation("relu", name="relu0")(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="pooling0")(x)

    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = init_filters * (2**stage)
            if block == 0 and stage == 0:
                x = conv_block_fn(filters, stage, block, strides=(1, 1))(x)
            elif block == 0:
                x = conv_block_fn(filters, stage, block, strides=(2, 2))(x)
            else:
                x = identity_block_fn(filters, stage, block)(x)

    x = BatchNormalization(name="bn1", **bn_params)(x)
    x = Activation("relu", name="relu1")(x)

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


def ResNet18(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights is not None:
        raise NotImplementedError("Weights loading is not implemented for ResNet18.")
    model = build_resnet(
        input_tensor=input_tensor,
        input_shape=input_shape,
        repetitions=(2, 2, 2, 2),
        classes=classes,
        include_top=include_top,
        block_type="basic",
    )
    model.name = "resnet18"
    return model


def ResNet34(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights is not None:
        raise NotImplementedError("Weights loading is not implemented for ResNet34.")
    model = build_resnet(
        input_tensor=input_tensor,
        input_shape=input_shape,
        repetitions=(3, 4, 6, 3),
        classes=classes,
        include_top=include_top,
        block_type="basic",
    )
    model.name = "resnet34"
    return model


def ResNet50(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights is not None:
        raise NotImplementedError("Weights loading is not implemented for ResNet50.")
    model = build_resnet(
        input_tensor=input_tensor,
        input_shape=input_shape,
        repetitions=(3, 4, 6, 3),
        classes=classes,
        include_top=include_top,
    )
    model.name = "resnet50"
    return model


def ResNet101(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights is not None:
        raise NotImplementedError("Weights loading is not implemented for ResNet101.")
    model = build_resnet(
        input_tensor=input_tensor,
        input_shape=input_shape,
        repetitions=(3, 4, 23, 3),
        classes=classes,
        include_top=include_top,
    )
    model.name = "resnet101"
    return model


def ResNet152(input_shape=None, input_tensor=None, weights=None, classes=1000, include_top=True):
    if weights is not None:
        raise NotImplementedError("Weights loading is not implemented for ResNet152.")
    model = build_resnet(
        input_tensor=input_tensor,
        input_shape=input_shape,
        repetitions=(3, 8, 36, 3),
        classes=classes,
        include_top=include_top,
    )
    model.name = "resnet152"
    return model
