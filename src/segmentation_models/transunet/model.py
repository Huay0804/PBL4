"""TransUNet-style segmentation model (Keras 3 / TensorFlow backend).

Architecture
------------
* CNN encoder: four conv blocks, each followed by 2x2 max-pooling, producing the
  skip features used by the decoder (matching the project U-Net pattern).
* CNN bottleneck: one conv block at 1/16 resolution.
* Patch embedding: a strided Conv2D tokenizes the bottleneck at 1/32 resolution
  only, keeping the attention sequence short (16x32 = 512 tokens for a
  512x1024 input) so it stays within an 8 GB GPU budget.
* Transformer: a stack of pre-norm encoder blocks over the tokens.
* Decoder: the Transformer output is reshaped back to a feature map, upsampled,
  and fused with the CNN skips through U-Net decoder blocks.
* Head: Conv2D(classes, 1x1) + softmax/sigmoid activation.

The model is image-only (a single input tensor); it does not consume bounding
box prior maps.
"""

from keras import layers
from keras.models import Model

from .blocks import (
    AddPositionEmbedding,
    conv_block,
    decoder_block,
    transformer_block,
)


# Encoder/decoder widths and Transformer hyper-parameters live here (not as CLI
# args) so the training protocol stays reproducible. They are tuned for an 8 GB
# GPU at 512x1024 with batch size 1.
_ENCODER_FILTERS = (64, 128, 256, 512)
_BOTTLENECK_FILTERS = 512
_DECODER_FILTERS = (256, 128, 64, 32)
_PATCH_SIZE = 2
_EMBED_DIM = 256
_TRANSFORMER_DEPTH = 4
_NUM_HEADS = 8
_MLP_DIM = 512
_DROPOUT = 0.1


def TransUNet(
    input_shape=(None, None, 3),
    classes=1,
    activation="softmax",
    embed_dim=_EMBED_DIM,
    depth=_TRANSFORMER_DEPTH,
    num_heads=_NUM_HEADS,
    mlp_dim=_MLP_DIM,
    patch_size=_PATCH_SIZE,
    dropout=_DROPOUT,
):
    inputs = layers.Input(shape=input_shape, name="data")
    x = inputs
    skips = []

    # CNN encoder: collect skip features before each downsample.
    for idx, filters in enumerate(_ENCODER_FILTERS, start=1):
        x = conv_block(x, filters, name=f"enc{idx}")
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{idx}")(x)

    # CNN bottleneck at 1/16 resolution.
    x = conv_block(x, _BOTTLENECK_FILTERS, name="bottleneck")

    # Patch embedding tokenizes the bottleneck at 1/32 resolution only.
    patches = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="patch_embed",
    )(x)
    feature_height = patches.shape[1]
    feature_width = patches.shape[2]
    if feature_height is None or feature_width is None:
        raise ValueError(
            "TransUNet requires a static spatial input_shape so the token grid "
            "is known at build time (e.g. (512, 1024, 3))."
        )

    tokens = layers.Reshape(
        (feature_height * feature_width, embed_dim), name="tokens"
    )(patches)
    tokens = AddPositionEmbedding(name="pos_embed")(tokens)
    tokens = layers.Dropout(dropout, name="embed_drop")(tokens)

    # Transformer encoder.
    for block_idx in range(depth):
        tokens = transformer_block(
            tokens,
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"transformer{block_idx + 1}",
        )
    tokens = layers.LayerNormalization(epsilon=1e-6, name="encoder_norm")(tokens)

    # Reshape tokens back to a feature map and restore the 1/16 resolution.
    x = layers.Reshape((feature_height, feature_width, embed_dim), name="tokens_grid")(tokens)
    x = layers.Conv2DTranspose(
        _DECODER_FILTERS[0],
        kernel_size=patch_size,
        strides=patch_size,
        padding="same",
        name="patch_expand",
    )(x)
    x = conv_block(x, _DECODER_FILTERS[0], name="proj")

    # U-Net decoder fusing the CNN skips.
    for idx in range(len(_ENCODER_FILTERS), 0, -1):
        x = decoder_block(
            x,
            skips[idx - 1],
            _DECODER_FILTERS[len(_ENCODER_FILTERS) - idx],
            name=f"dec{idx}",
        )

    x = layers.Conv2D(classes, (1, 1), padding="same", name="final_conv")(x)
    # Force float32 logits/probabilities so mixed precision stays numerically
    # stable and the saved output dtype is consistent across policies.
    x = layers.Activation(activation, dtype="float32", name=activation)(x)
    model = Model(inputs, x, name="transunet")
    return model
