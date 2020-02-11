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
import tensorflow as tf
from functools import partial
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    multiply,
    ReLU)
from keras.layers import Input
from keras.initializers import he_uniform
from keras.models import Model
# pylint: enable=import-error


import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model


import os.path
import conversion
import data
import os


# 定义一个函数，path为你的路径
def traversalDir_FirstDir(path):
    # path 输入是 Dev/Test
    # 定义一个字典，用来存储结果————歌名：路径
    dict = {}
    # 获取该目录下的所有文件夹目录, 每个歌的文件夹

    files = os.listdir(path)
    for file in files:
        # 得到该文件下所有目录的路径
        m = os.path.join(path, file)
        h = os.path.split(m)
        dict[h[1]] = []
        song_wav = os.listdir(m)
        m = m + '/'
        for track in song_wav:
            value = os.path.join(m, track)
            dict[h[1]].append(value)
    return dict

mix_path = traversalDir_FirstDir('DSD100subset/Mixtures/Dev/')

sou_path = traversalDir_FirstDir('DSD100subset/Sources/Dev/')

all_path = mix_path.copy()
for key in all_path.keys():
    all_path[key].extend(sou_path[key])

print(all_path)


def preprocessing(all_path_para=all_path, fftWindowSize=1536, SLICE_SIZE=128):
    x = []
    y = []

    for key in all_path_para:
        path_list = [all_path[key][0], all_path[key][-1]]

        for path in path_list:
            audio, sampleRate = conversion.loadAudioFile(path)
            spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize)
            print(spectrogram.shape)

            # chop into slices so everything's the same size in a batch 切为模型输入尺寸
            dim = SLICE_SIZE
            Slices = data.chop(spectrogram, dim)   # 114个128*128
            print(len(Slices))
            if 'mixture' in path:
                x.extend(Slices)
            else:
                y.extend(Slices)

    x = np.array(x)[:, :, :, np.newaxis]
    y = np.array(y)[:, :, :, np.newaxis]

    x = np.asarray(x)
    y = np.asarray(y)
    return [x,y]


[x,y] = preprocessing(all_path_para=all_path, fftWindowSize=1536, SLICE_SIZE=128)
print(x.shape)


def train(trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])


def valid(trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])


xTrain, yTrain = train()
xValid, yValid = valid()



def apply_unet():
    """ Apply a convolutionnal U-net to model a single instrument (one U-net
    is used for each instrument).

    :param input_tensor:
    :param output_name: (Optional) , default to 'output'
    :param params: (Optional) , default to empty dict.
    :param output_mask_logit: (Optional) , default to False.
    """
    conv_n_filters = [16, 32, 64, 128, 256, 512]
    conv_activation_layer = ReLU()
    deconv_activation_layer = ReLU()
    kernel_initializer = he_uniform(seed=50)
    conv2d_factory = partial(
        Conv2D,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)

    input_tensor = Input(shape=(None, None, 1), name='input')

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
    batch6 = BatchNormalization(axis=-1)(conv6)
    _ = conv_activation_layer(batch6)
    #
    #
    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))((conv6))
    up1 = deconv_activation_layer(up1)
    batch7 = BatchNormalization(axis=-1)(up1)
    drop1 = Dropout(0.5)(batch7)
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
    up7 = Conv2D(1, (4, 4), dilation_rate=(2, 2), activation='sigmoid', padding='same', kernel_initializer=kernel_initializer)((batch12))
    output = multiply([up7, input_tensor])

    m = Model(inputs=input_tensor, outputs=output)

    m.compile(loss='mean_squared_error', optimizer='adam')

    model = m

    model.fit(xTrain, yTrain, batch_size=30, epochs=10, validation_data=(xValid, yValid))
    weightPath = 'C:\\Users\\chaow\\PycharmProjects\\Akabot\\outweight' + ".h5"
    model.save_weights(weightPath, overwrite=True)


apply_unet()

