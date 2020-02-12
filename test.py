import pandas as pd

#dsd = pd.read_excel('C:/Users/chaow/PycharmProjects/Akabot/DSD100subset/dsd100.xlsx')
#print(dsd.Style.value_counts())

import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model

#path_list = ['DSD100subset/Mixtures/Dev/055/mixture.wav', 'DSD100subset/Sources/Dev/055/vocals.wav']

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
    return [x,y]


[x,y] = preprocessing(all_path_para=all_path, fftWindowSize=1536, SLICE_SIZE=128)
print(x.shape)


def train(trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])


def valid(trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])


def trainModel(epochs=1, batch=8):
    mashup = Input(shape=(None, None, 1), name='input')  # shape不含batch size, None意味着可以随便取
    convA = Conv2D(64, 3, activation='relu', padding='same')(mashup)  # 64个filter, 每个都是3*3, 输入padding加0至输出大小等于输入
    conv = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(convA)  # 默认True, 但这层不用bias vector
    conv = BatchNormalization()(conv)

    convB = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 4, strides=2, activation='relu', padding='same', use_bias=False)(convB)
    conv = BatchNormalization()(conv)  # 不改变尺寸

    conv = Conv2D(128, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(128, 3, activation='relu', padding='same', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)  # 行重复2次，列重复2次。默认channels_last (default)：(batch, height, width, channels)   为什么大小？

    conv = Concatenate()([conv, convB])  # 默认沿axis=-1
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 3, activation='relu', padding='same', use_bias=False)(conv)
    conv = BatchNormalization()(conv)
    conv = UpSampling2D((2, 2))(conv)

    conv = Concatenate()([conv, convA])
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(64, 3, activation='relu', padding='same')(conv)

    conv = Conv2D(32, 3, activation='relu', padding='same')(conv)
    conv = Conv2D(1, 3, activation='relu', padding='same')(conv)  # 输出只1channel
    acapella = conv

    m = Model(inputs=mashup, outputs=acapella)
    m.compile(loss='mean_squared_error', optimizer='adam')
    model = m

    xTrain, yTrain = train()
    xValid, yValid = valid()

    model.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))
    weightPath = 'D:\\AAA\\CS\\pole\\Akabot\\outweight' + ".h5"
    model.save_weights(weightPath, overwrite=True)

#trainModel()

