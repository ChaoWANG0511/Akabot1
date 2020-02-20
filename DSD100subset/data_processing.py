import numpy as np
import os
import conversion
from keras.layers import Input, Conv2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model
from math import ceil

#trying to train a model with the original method, but with as input rectangle slices with overlaps.


def listdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def traversalDir_FirstDir(path):
    dict = {}
    files = listdirInMac(path)
    for file in files:
        m = os.path.join(path, file)
        h = os.path.split(m)
        dict[h[1]] = []
        song_wav = listdirInMac(m)
        m = m + '/'
        for track in song_wav:
            value = os.path.join(m, track)
            dict[h[1]].append(value)
    return dict

mix_path = traversalDir_FirstDir('Mixtures/Test/')

sou_path = traversalDir_FirstDir('Sources/Test/')

all_path = mix_path.copy()
for key in all_path.keys():
    all_path[key].extend(sou_path[key])

print(mix_path,"\n\n %d mixtures in the given path to train"%len(mix_path))

def chop(matrix, time_scale, ratio_overlap):
    slices = []
    for time in range(0, ceil((matrix.shape[1]-time_scale)/time_scale/(1-ratio_overlap))+1):   # 列
            s = matrix[: (int(matrix.shape[0]/64))*64,
                       int(time * time_scale*(1-ratio_overlap)) :int(time * time_scale*(1-ratio_overlap))+time_scale ]
            print(s.shape)
            slices.append(s)
    return slices

def expandToGrid(spectrogram, time_scale, ratio_overlap):
    # crop along both axes
    newY = int(ceil((spectrogram.shape[1] - time_scale)/(1-ratio_overlap)/time_scale) * (1-ratio_overlap)*time_scale)+time_scale  # ceil取上
    newX = spectrogram.shape[0]
    newSpectrogram = np.zeros((newX, newY))   # 右，下：多一些0，至能被girdsize整除
    newSpectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram
    return newSpectrogram

def preprocessing(all_path_para, fftWindowSize=1536, time_scale=256, ratio_overlap=0.2):
    x = []
    y = []
    num_sli=0
    for key in all_path_para:
        path_list = [all_path_para[key][0], all_path_para[key][-1]]

        for path in path_list:
            audio, sampleRate = conversion.loadAudioFile(path)
            spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize)
            print("original size of the audio:",spectrogram.shape)

            # chop into slices so everything's the same size in a batch 切为模型输入尺寸
            dim = time_scale

            newspectrogram=expandToGrid(spectrogram, time_scale, ratio_overlap)
            Slices = chop(newspectrogram, dim, ratio_overlap)
            if 'mixture' in path:
                x.extend(Slices)
            else:
                y.extend(Slices)
        num_sli = num_sli + len(Slices)

    print("there are in total %d slices to train" %num_sli )

    x = np.array(x)[:, :, :, np.newaxis]
    y = np.array(y)[:, :, :, np.newaxis]
    return [x,y]

[x,y] = preprocessing(all_path_para=all_path)

def train(trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])


def valid(trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])


def trainModel(epochs=3, batch=8):
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
    weightPath = '..\weights' + ".h5"
    model.save_weights(weightPath, overwrite=True)


trainModel()
