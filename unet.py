#!/usr/bin/env python
# coding: utf8

"""
This module contains building functions for U-net source
separation models in a similar way as in A. Jansson et al. "Singing
voice separation with deep u-net convolutional networks", ISMIR 2017.
Each instrument is modeled by a single U-net convolutional
/ deconvolutional network that take a mix spectrogram as input and the
estimated sound spectrogram as output.
"""

# pylint: disable=import-error
from functools import partial

from keras.backend import int_shape
from keras.backend import squeeze
from keras.initializers import he_uniform
from keras.layers import Concatenate
from keras.layers import (
    Conv2DTranspose,
    Dropout,
    multiply,
    ReLU)
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def apply_unet():
    conv_n_filters = [16, 32, 64, 128, 256, 512]
    conv_activation_layer = ReLU()
    deconv_activation_layer = ReLU()
    kernel_initializer = he_uniform(seed=50)
    conv2d_factory = partial(
        Conv2D,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)

    input_tensor = Input(batch_shape=(1, 768, 256, 1), name='input')
    print(input_tensor.shape)
    # First layer.
    conv1 = conv2d_factory(conv_n_filters[0], (5, 5))(input_tensor)
    batch1 = BatchNormalization(axis=-1)(conv1)
    rel1 = conv_activation_layer(batch1)
    # Second layer.
    conv2 = conv2d_factory(conv_n_filters[1], (5, 5))(rel1)
    batch2 = BatchNormalization(axis=-1)(conv2)
    rel2 = conv_activation_layer(batch2)
    # Third layer.
    conv3 = conv2d_factory(conv_n_filters[2], (5, 5))(rel2)
    batch3 = BatchNormalization(axis=-1)(conv3)
    rel3 = conv_activation_layer(batch3)
    # Fourth layer.
    conv4 = conv2d_factory(conv_n_filters[3], (5, 5))(rel3)
    batch4 = BatchNormalization(axis=-1)(conv4)
    rel4 = conv_activation_layer(batch4)
    # Fifth layer.
    conv5 = conv2d_factory(conv_n_filters[4], (5, 5))(rel4)
    batch5 = BatchNormalization(axis=-1)(conv5)
    rel5 = conv_activation_layer(batch5)
    # Sixth layer
    conv6 = conv2d_factory(conv_n_filters[5], (5, 5))(rel5)

    print(1, int_shape(conv1))
    print(5, int_shape(conv5))
    print(6, int_shape(conv6))

    conv_to_LSTM_dims = int_shape(conv6)
    x = Reshape(target_shape=conv_to_LSTM_dims, name='reshapeconvtolstm')(conv6)

    x = ConvLSTM2D(filters=32, kernel_size=(3, 3),
                   padding='same', return_sequences=True, stateful=True)(x)

    x = ConvLSTM2D(filters=conv_n_filters[5], kernel_size=(3, 3),
                   padding='same', return_sequences=True, stateful=True)(x)

    print(int_shape(x))

    LSTM_to_conv_dims = int_shape(conv6)
    x = Reshape(target_shape=LSTM_to_conv_dims, name='reshapelstmtoconv')(x)
    x = Lambda(lambda y: squeeze(y, 0))(x)
    print('将进酒', int_shape(x))

    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))(x)
    up1 = deconv_activation_layer(up1)
    batch7 = BatchNormalization(axis=-1)(up1)
    drop1 = Dropout(0.5)(batch7)
    print(int_shape(conv5))
    print(int_shape(drop1))
    merge1 = Concatenate(axis=-1)([conv5, drop1])
    #
    up2 = conv2d_transpose_factory(conv_n_filters[3], (5, 5))((merge1))
    up2 = deconv_activation_layer(up2)
    batch8 = BatchNormalization(axis=-1)(up2)
    drop2 = Dropout(0.5)(batch8)
    merge2 = Concatenate(axis=-1)([conv4, drop2])
    #
    up3 = conv2d_transpose_factory(conv_n_filters[2], (5, 5))((merge2))
    up3 = deconv_activation_layer(up3)
    batch9 = BatchNormalization(axis=-1)(up3)
    drop3 = Dropout(0.5)(batch9)
    merge3 = Concatenate(axis=-1)([conv3, drop3])
    #
    up4 = conv2d_transpose_factory(conv_n_filters[1], (5, 5))((merge3))
    up4 = deconv_activation_layer(up4)
    batch10 = BatchNormalization(axis=-1)(up4)
    merge4 = Concatenate(axis=-1)([conv2, batch10])
    #
    up5 = conv2d_transpose_factory(conv_n_filters[0], (5, 5))((merge4))
    up5 = deconv_activation_layer(up5)
    batch11 = BatchNormalization(axis=-1)(up5)
    merge5 = Concatenate(axis=-1)([conv1, batch11])
    #
    up6 = conv2d_transpose_factory(1, (5, 5), strides=(2, 2))((merge5))
    up6 = deconv_activation_layer(up6)
    batch12 = BatchNormalization(axis=-1)(up6)
    # Last layer to ensure initial shape reconstruction.
    up7 = Conv2D(1, (4, 4), dilation_rate=(2, 2), activation='sigmoid', padding='same',
                 kernel_initializer=kernel_initializer)((batch12))
    output = multiply([up7, input_tensor])
    print(input_tensor.shape, output.shape)

    m = Model(inputs=input_tensor, outputs=output)
    return m


