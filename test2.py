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


xTrain, yTrain = train()
xValid, yValid = valid()

#blstm
def train_blstm(input_tensor, output_name='output'):
    units = 250
    kernel_initializer = he_uniform(seed=50)
    flatten_input = TimeDistributed(Flatten())((input_tensor))

    def create_bidirectional():
        return Bidirectional(
            CuDNNLSTM(
                units,
                kernel_initializer=kernel_initializer,
                return_sequences=True))

    l1 = create_bidirectional()((flatten_input))
    l2 = create_bidirectional()((l1))
    l3 = create_bidirectional()((l2))
    dense = TimeDistributed(
        Dense(
            int(flatten_input.shape[2]),
            activation='relu',
            kernel_initializer=kernel_initializer))((l3))
    output = TimeDistributed(
        Reshape(input_tensor.shape[2:]),
        name=output_name)(dense)

    return output
#output = train_blstm(xTrain,output_name='output', params={})
#print(output.shape)
#print(yTrain.shape)

output = train_blstm(xTrain, output_name='output')
cross_entropy = -tf.reduce_mean(yTrain * tf.log(tf.clip_by_value(output, 1e-10, 1.0)))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cross_entropy)  # Adam Optimizer


sess = tf.Session()
sess.run(tf.initialize_all_variables())


def testing_func(testing_data, testing_label, sess):
    loss_ = sess.run([loss], feed_dict={X_: testing_data, Y: testing_label})
    print('loss: ', loss_)


print()
print("开始训练")
for i in range(20):
    print()
    print("训练轮数: ", i)
    _, loss_ = sess.run([optimizer, loss], feed_dict={X_: xTrain, Y: yTrain})

    print("#######################################")
    print("testing: ")
    print("training集: ")
    testing_func(xTrain, yTrain, sess)
    print("#######################################")

    print("validation集: ")
    testing_func(xValid, yValid, sess)
    print("#######################################")

