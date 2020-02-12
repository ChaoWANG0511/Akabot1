#!/usr/bin/env python
# -*- coding:utf-8 -*-


import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model

#path_list = ['DSD100subset/Mixtures/Dev/055/mixture.wav', 'DSD100subset/Sources/Dev/055/vocals.wav']

import os.path
import conversion
import data
import os

#blstm
import tensorflow as tf
from tensorflow.compat.v1.keras.initializers import he_uniform
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Flatten,
    Reshape,
    TimeDistributed)

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

    #x = np.array(x)[:, :, :, np.newaxis]
    #y = np.array(y)[:, :, :, np.newaxis]
    x = np.array(x)
    y = np.array(y)
    return [x,y]


[x,y] = preprocessing(all_path_para=all_path, fftWindowSize=1536, SLICE_SIZE=128)
#print(x.shape)


def train(trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])


def valid(trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])


#blstm
def trainModel(epochs=1, batch=8):
    mashup = Input(shape=(None, None), name='input')  # shape不含batch size, None意味着可以随便取
    units = 250
    kernel_initializer = he_uniform(seed=50)
    flatten_input = TimeDistributed(Flatten())((mashup))

    def create_bidirectional():
        return Bidirectional(
            CuDNNLSTM(
                units,
                kernel_initializer=kernel_initializer,
                return_sequences=True))

    l1 = create_bidirectional()((mashup))
    l2 = create_bidirectional()((l1))
    l3 = create_bidirectional()((l2))
    dense = TimeDistributed(
        Dense(
            int(mashup.shape[2]),
            activation='relu',
            kernel_initializer=kernel_initializer))((l3))
    output = TimeDistributed(
        Reshape(mashup.shape[2:]),
        name=output_name)(dense)
    acapella = output
    m = Model(inputs=mashup, outputs=acapella)
    m.compile(loss='mean_squared_error', optimizer='adam')
    model = m

    xTrain, yTrain = train()
    xValid, yValid = valid()

    model.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))
    weightPath = 'D:\\AAA\\CS\\pole\\Akabot\\outweightlstm' + ".h5"
    model.save_weights(weightPath, overwrite=True)



