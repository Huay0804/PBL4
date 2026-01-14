import numpy as np
import tensorflow as tf

def transpose_shape(shape, target_format, spatial_axes):
    if target_format=='channels_first':
        new_values=shape[:spatial_axes[0]]
        new_values+=(shape[-1],)
        new_values+=tuple(shape[x] for x in spatial_axes)

        if isinstance(shape, list):
            return list(new_values)
        return new_values
    elif target_format=='channels_last':
        return shape
    else:
        raise ValueError("The 'data_format' argument must be one of 'channels_first', 'channels_last'. Received: " + str(target_format))
    
def permute_dimensions(x, pattern):
    return tf.transpose(x, perm=pattern)

def int_shape(x):
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    shape = getattr(x, "shape", None)
    if shape is not None:
        if hasattr(shape, "as_list"):
            return tuple(shape.as_list())
        try:
            return tuple(shape)
        except TypeError:
            pass
    try:
        return tuple(x.get_shape().as_list())
    except (AttributeError, ValueError):
        return None
    
def resize_images(x, 
                  height_factor,
                  width_factor,
                  data_format,
                  interpolation='nearest'):
    if data_format == 'channels_first':
        rows, cols = 2, 3
    else: 
        rows, cols = 1, 2

    original_shape = int_shape(x)
    new_shape = tf.shape(x)[rows:cols+1]
    new_shape = new_shape * tf.constant(np.array([height_factor, width_factor]), dtype='int32')

    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
    if interpolation == 'nearest':
        x = tf.image.resize(x, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    elif interpolation == 'bilinear':
        x = tf.image.resize(x, new_shape, method=tf.image.ResizeMethod.BILINEAR)
    else:
        raise ValueError("interpolation should be one of 'nearest' or 'bilinear'.")
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 3, 1, 2])
    
    if original_shape is not None:
        if original_shape[rows] is None:
            new_height = None
        else:
            new_height = original_shape[rows] * height_factor

        if original_shape[cols] is None:
            new_width = None
        else:
            new_width = original_shape[cols] * width_factor

        output_shape = (None, new_height, new_width, None)
        x = tf.ensure_shape(
            x,
            transpose_shape(output_shape, data_format, spatial_axes=(1, 2)),
        )
    return x
