from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import BatchNormalization

def Conv2DBlock(n_filters,
                kernel_size,
                activation='relu',
                use_batchNorm='True',
                name='conv_block',
                **kwargs
):
    