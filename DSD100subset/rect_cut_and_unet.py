import numpy as np
import os
import conversion
from math import ceil
from unet import apply_unet


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

mix_path = traversalDir_FirstDir('C:/Users/jy/Desktop/to train_new/Mixtures/')

sou_path = traversalDir_FirstDir('C:/Users/jy/Desktop/to train_new/Sources/')

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
            slices.append(s)  # 切为很多小块，每块scale*scale，存到slices列表， 使得模型的输入都一样大
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
            newspectrogram=expandToGrid(spectrogram, time_scale, ratio_overlap)
            Slices = chop(newspectrogram, time_scale, ratio_overlap)
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


def trainModel(epochs=1, batch=1):

    m=apply_unet()

    m.compile(loss='mean_squared_error', optimizer='adam')

    model = m

    xTrain, yTrain = train()

    xValid, yValid = valid()

    model.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))
    weightPath = '../weights_kkk' + ".h5"
    model.save_weights(weightPath, overwrite=True)

trainModel()
