from xnet.model import Xnet

# prepare data
import os
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

mix_path = traversalDir_FirstDir('DSD100subset/Mixtures/Dev')

sou_path = traversalDir_FirstDir('DSD100subset/Sources/Dev')

all_path = mix_path.copy()
for key in all_path.keys():
    all_path[key].extend(sou_path[key])

print("%d mixtures in the given path to train"%len(mix_path))

import librosa
import numpy as np
def loadAudioFile(filePath):
    audio, sampleRate = librosa.load(filePath)  # Load an audio file as a floating point time series， 默认采样率sr=22050
    return audio, sampleRate

# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioFile, fftWindowSize):
    spectrogram = librosa.stft(audioFile, fftWindowSize)   # 返回复数值矩阵D , STFT矩阵D中的行数是（1 + 第二个参数一般2的幂n_fft / 2）
    phase = np.angle(spectrogram)
    amplitude = np.log1p(np.abs(spectrogram))   # np.abs(D[f, t]) is the magnitude of frequency bin f at frame帧 t， loge(1+ )
    return amplitude, phase

from math import ceil
def expandToGrid_unet(spectrogram, time_scale, ratio_overlap):
    newY = int(ceil((spectrogram.shape[1] - time_scale)/(1-ratio_overlap)/time_scale) * (1-ratio_overlap)*time_scale)+time_scale  # ceil取上
    #newX = int(spectrogram.shape[0]/64)*64
    newX = 512
    newSpectrogram = np.zeros((newX, newY))   # 右，下：多一些0，至能被girdsize整除
    print(spectrogram.shape,"original shape")
    print(newSpectrogram.shape,"expanded shape")
    newSpectrogram[:512, :spectrogram.shape[1]] = spectrogram[:512,:]
    return newSpectrogram

def chop_unet(matrix, time_scale, ratio_overlap):
    slices = []
    for time in range(0, ceil((matrix.shape[1]-time_scale)/time_scale/(1-ratio_overlap))+1):   # 列
            s = matrix[: (int(matrix.shape[0]/64))*64,
                       int(time * time_scale*(1-ratio_overlap)) :int(time * time_scale*(1-ratio_overlap))+time_scale ]
            slices.append(s)
    print(s.shape)
    return slices


def song2spectrogram(all_path_para, fftWindowSize=1024, time_scale=64, overlap_ratio=0.5):
    x = np.empty((0, 512, 64), float)
    x_phase = []
    y_bass = np.empty((0, 512, 64), float)
    y_drums = np.empty((0, 512, 64), float)
    y_other = np.empty((0, 512, 64), float)
    y_vocals = np.empty((0, 512, 64), float)
    for key in all_path_para:
        path_list = all_path_para[key][:]

        for path in path_list:
            audio, sampleRate = loadAudioFile(path)
            print('sample rate: ', sampleRate)
            spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize)
            print("Spectrogram shape:", spectrogram.shape)

            expandedSpectrogram = expandToGrid_unet(spectrogram, time_scale, overlap_ratio)
            slices = chop_unet(expandedSpectrogram, time_scale, overlap_ratio)

            if 'mixture' in path:
                x = np.append(x, slices, axis=0)
                x_phase.append(phase)
            elif 'bass' in path:
                y_bass = np.append(y_bass, slices, axis=0)
            elif 'drums' in path:
                y_drums = np.append(y_drums, slices, axis=0)
            elif 'other' in path:
                y_other = np.append(y_other, slices, axis=0)
            else:
                y_vocals = np.append(y_vocals, slices, axis=0)

    output = [x, x_phase, y_bass, y_drums, y_other, y_vocals]

    return output


list_dataset = song2spectrogram(all_path_para=all_path)
x = list_dataset[0]
y = list_dataset[-1]

x1 = np.zeros( (  x.shape[0], x.shape[1],  x.shape[2], 1 ) )
x1[:,:,:,0] = x
#x1[:,:,:,1] = x
#x1[:,:,:,2] = x

y1 = np.zeros( (  y.shape[0], y.shape[1],  y.shape[2], 1 ) )
y1[:,:,:,0] = y
#y1[:,:,:,1] = y
#y1[:,:,:,2] = y
print(x1.shape, y1.shape)  # (231, 512, 64, 3) (231, 512, 64, 1)
#print(x1.max(), y1.max())





#x, y = ... # range in [0,1], the network expects input channels of 3

# prepare model

#model = Xnet(backbone_name='resnet50', encoder_weights='imagenet', decoder_block_type='transpose') # build UNet++

#model.summary()

#model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# train model
#model.fit(x1, y1)

from helper import Nest_Net

model = Nest_Net(512,64,1)
model.summary()
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(x1, y1, batch_size=None, epochs=2)   # 一轮：loss: 0.9725 - accuracy: 0.0017
