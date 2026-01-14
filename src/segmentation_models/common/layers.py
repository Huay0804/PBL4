import tensorflow as tf

from keras.layers import Layer
from keras.layers import InputSpec
from keras.utils import get_custom_objects

from .functions import resize_images


def _normalize_tuple(value, n, name):
    """Minimal replacement for conv_utils.normalize_tuple (Keras 2)."""
    if isinstance(value, int):
        return (value,) * n
    if isinstance(value, (tuple, list)) and len(value) == n:
        return tuple(int(v) for v in value)
    raise ValueError(f"The `{name}` argument must be an int or a tuple/list of {n} ints. Received: {value!r}")


class ResizeImage(Layer):
    """ResizeImage layer for 2D inputs (rank-4)."""

    def __init__(self, factor=(2, 2), data_format='channels_last', interpolation='nearest', **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.factor = _normalize_tuple(factor, 2, 'factor')
        self.input_spec = InputSpec(ndim=4)

        if interpolation not in ('nearest', 'bilinear'):
            raise ValueError('interpolation should be one of "nearest" or "bilinear".')
        self.interpolation = interpolation

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.factor[0] * input_shape[2] if input_shape[2] is not None else None
            width  = self.factor[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0], input_shape[1], height, width)
        else:  # channels_last
            height = self.factor[0] * input_shape[1] if input_shape[1] is not None else None
            width  = self.factor[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        return resize_images(inputs, self.factor[0], self.factor[1], self.data_format, self.interpolation)

    def get_config(self):
        config = {
            'factor': self.factor,
            'data_format': self.data_format,
            'interpolation': self.interpolation,
        }
        base_config = super().get_config()
        return {**base_config, **config}


get_custom_objects().update({'ResizeImage': ResizeImage})
