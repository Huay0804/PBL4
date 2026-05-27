"""Building blocks for the TransUNet-style segmentation model.

The design follows the original TransUNet idea (Chen et al., 2021): a CNN
encoder produces multi-scale skip features, a Transformer models global context
over a tokenized low-resolution feature map, and a U-Net decoder fuses the
Transformer output with the CNN skips. The implementation is intentionally
lightweight so it fits an 8 GB GPU at 512x1024 input with batch size 1.
"""

import keras
from keras import layers


def conv_block(x, filters, name):
    """Two 3x3 conv + BN + ReLU layers, matching the project U-Net style."""
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def decoder_block(x, skip, filters, name):
    """Transposed-conv upsample, concat the CNN skip, then a conv block."""
    x = layers.Conv2DTranspose(
        filters, (2, 2), strides=(2, 2), padding="same", name=f"{name}_up"
    )(x)
    if skip is not None:
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = conv_block(x, filters, name=f"{name}_conv")
    return x


@keras.saving.register_keras_serializable(package="transunet")
class AddPositionEmbedding(layers.Layer):
    """Adds a learnable positional embedding to a sequence of tokens.

    Registered for serialization so that ``keras.models.load_model`` can rebuild
    saved TransUNet checkpoints (the segmentation_models package import triggers
    registration).
    """

    def build(self, input_shape):
        num_tokens = int(input_shape[1])
        dim = int(input_shape[2])
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, num_tokens, dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + keras.ops.cast(self.position_embedding, inputs.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape


def transformer_block(x, num_heads, key_dim, mlp_dim, dropout, name):
    """Pre-norm Transformer encoder block (MHA + MLP) with residual paths."""
    embed_dim = x.shape[-1]

    attn_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=f"{name}_mha",
    )(attn_norm, attn_norm)
    x = layers.Add(name=f"{name}_attn_add")([x, attn])

    mlp_norm = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    mlp = layers.Dense(mlp_dim, activation="gelu", name=f"{name}_mlp_dense1")(mlp_norm)
    mlp = layers.Dropout(dropout, name=f"{name}_mlp_drop1")(mlp)
    mlp = layers.Dense(embed_dim, name=f"{name}_mlp_dense2")(mlp)
    mlp = layers.Dropout(dropout, name=f"{name}_mlp_drop2")(mlp)
    x = layers.Add(name=f"{name}_mlp_add")([x, mlp])
    return x
