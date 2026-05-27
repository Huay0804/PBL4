from keras.layers import Input

from .builder import build_nestnet


_ICPR_ENCODER_FILTERS = (64, 128, 256, 512, 512)
_ICPR_DECODER_FILTERS = (512, 256, 128, 64)
_ICPR_UPSAMPLE_RATES = (2, 2, 2, 2)


def Nestnet(
    input_shape=(None, None, 3),
    input_tensor=None,
    encoder_weights=None,
    freeze_encoder=False,
    skip_connections="default",
    decoder_block_type="upsampling",
    decoder_filters=None,
    decoder_use_batchnorm=True,
    n_upsample_blocks=4,
    upsample_rates=None,
    classes=1,
    activation="sigmoid",
    deep_supervision=True,
    deep_supervision_output_index=None,
):
    del encoder_weights, freeze_encoder, skip_connections

    if input_tensor is None:
        input_tensor = Input(shape=input_shape, name="image_input")

    model = build_nestnet(
        classes=classes,
        input_tensor=input_tensor,
        encoder_filters=_ICPR_ENCODER_FILTERS,
        decoder_filters=decoder_filters or _ICPR_DECODER_FILTERS,
        block_type=decoder_block_type,
        activation=activation,
        n_upsample_blocks=n_upsample_blocks,
        upsample_rates=upsample_rates or _ICPR_UPSAMPLE_RATES,
        use_batchnorm=decoder_use_batchnorm,
        deep_supervision=deep_supervision,
        deep_supervision_output_index=deep_supervision_output_index,
    )
    model.name = "nest-icpr"
    return model


def ModifiedNestnet(
    input_shape=(None, None, 3),
    input_tensor=None,
    encoder_weights=None,
    freeze_encoder=False,
    skip_connections="default",
    decoder_block_type="upsampling",
    decoder_filters=None,
    decoder_use_batchnorm=True,
    n_upsample_blocks=4,
    upsample_rates=None,
    classes=1,
    activation="sigmoid",
    bb_channels=None,
    bb_use_batchnorm=True,
    deep_supervision=True,
    deep_supervision_output_index=None,
):
    del encoder_weights, freeze_encoder, skip_connections

    if input_tensor is None:
        input_tensor = Input(shape=input_shape, name="image_input")

    bb_channels = bb_channels or classes
    bb_input = Input(shape=input_shape[:2] + (bb_channels,), name="bb_input")

    model = build_nestnet(
        classes=classes,
        input_tensor=input_tensor,
        encoder_filters=_ICPR_ENCODER_FILTERS,
        decoder_filters=decoder_filters or _ICPR_DECODER_FILTERS,
        block_type=decoder_block_type,
        activation=activation,
        n_upsample_blocks=n_upsample_blocks,
        upsample_rates=upsample_rates or _ICPR_UPSAMPLE_RATES,
        use_batchnorm=decoder_use_batchnorm,
        bb_input=bb_input,
        bb_use_batchnorm=bb_use_batchnorm,
        deep_supervision=deep_supervision,
        deep_supervision_output_index=deep_supervision_output_index,
    )
    model.name = "modnest-icpr"
    return model
