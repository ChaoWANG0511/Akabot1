import numpy as np
import os
from math import ceil
import librosa
import skimage.io as io


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


def loadAudioFile(filePath):
    audio, sampleRate = librosa.load(filePath)  # Load an audio file as a floating point time series， 默认采样率sr=22050
    return audio, sampleRate


# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioFile, fftWindowSize):
    spectrogram = librosa.stft(audioFile, fftWindowSize)  # 返回复数值矩阵D , STFT矩阵D中的行数是（1 + 第二个参数一般2的幂n_fft / 2）
    phase = np.angle(spectrogram)
    amplitude = np.log1p(
        np.abs(spectrogram))  # np.abs(D[f, t]) is the magnitude of frequency bin f at frame帧 t， loge(1+ )
    return amplitude, phase


def expandToGrid(spectrogram, time_scale, ratio_overlap):
    # 条件： time_scale * ratio_overlap 是整数
    K = int(ceil((spectrogram.shape[1] / time_scale - ratio_overlap) / (1 - ratio_overlap)))  # ceil取上
    newY = int(time_scale + time_scale * (1 - ratio_overlap) * (K - 1))
    newX = spectrogram.shape[0]  # 行数不变
    newSpectrogram = np.zeros((newX, newY))  # 右：多一些0，至能被girdsize整除
    newSpectrogram[:, :spectrogram.shape[1]] = spectrogram
    return newSpectrogram, K


def expandToGrid_unet(spectrogram, time_scale, ratio_overlap):
    newY = int(ceil((spectrogram.shape[1] - time_scale) / (1 - ratio_overlap) / time_scale) * (
                1 - ratio_overlap) * time_scale) + time_scale  # ceil取上
    # newX = int(spectrogram.shape[0]/64)*64
    newX = 512
    newSpectrogram = np.zeros((newX, newY))  # 右，下：多一些0，至能被girdsize整除
    print(spectrogram.shape, "original shape")
    print(newSpectrogram.shape, "expanded shape")
    newSpectrogram[:512, :spectrogram.shape[1]] = spectrogram[:512, :]
    return newSpectrogram


def chop_unet(matrix, time_scale, ratio_overlap):
    slices = []
    for time in range(0, ceil((matrix.shape[1] - time_scale) / time_scale / (1 - ratio_overlap)) + 1):  # 列
        s = matrix[: (int(matrix.shape[0] / 64)) * 64,
            int(time * time_scale * (1 - ratio_overlap)):int(time * time_scale * (1 - ratio_overlap)) + time_scale]
        slices.append(s)
    print(s.shape)
    return slices


def chop(matrix, time_scale, ratio_overlap):
    slices = []
    for time in range(0, ceil((matrix.shape[1] - time_scale) / time_scale / (1 - ratio_overlap)) + 1):  # 列
        s = matrix[:, int(time * time_scale * (1 - ratio_overlap)):int(
            time * time_scale * (1 - ratio_overlap) + time_scale)]  # 切为很多小块，每块513*scale
        s = np.transpose(s)  # 每块scale*513
        slices.append(s)  # 存到slices列表， 使得模型的输入都一样大
    slices_np = np.array(slices)
    print(slices_np.shape)
    return slices_np


def song2spectrogram(all_path_para, fftWindowSize=1024, time_scale=30, overlap_ratio=0.5):
    x = np.empty((0, 30, 513), float)
    x_phase = []
    y_bass = np.empty((0, 30, 513), float)
    y_drums = np.empty((0, 30, 513), float)
    y_other = np.empty((0, 30, 513), float)
    y_vocals = np.empty((0, 30, 513), float)
    for key in all_path_para:
        path_list = all_path_para[key][:]

        for path in path_list:
            audio, sampleRate = loadAudioFile(path)
            print('sample rate: ', sampleRate)
            spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize)
            print("Spectrogram shape:", spectrogram.shape)

            expandedSpectrogram, K = expandToGrid(spectrogram, time_scale, overlap_ratio)
            slices = chop(expandedSpectrogram, time_scale, overlap_ratio)

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


def song2spectrogram_unet(all_path_para, fftWindowSize=1024, time_scale=64, overlap_ratio=0.5):
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


def ichop(expandedSpectrogram, predicted_slices, time_scale=30, ratio_overlap=0.5):
    newSpectrogram = np.zeros((expandedSpectrogram.shape[0], expandedSpectrogram.shape[1]))  # (513, 16110)
    for i in range(predicted_slices.shape[0]):  # 1073
        toInput = np.transpose(predicted_slices[i, :, :])  # (513, 30)
        if i == 0:
            newSpectrogram[:, i:i + time_scale] = toInput
        else:
            newSpectrogram[:, int(i * time_scale * (1 - ratio_overlap)):int(
                (i * time_scale * (1 - ratio_overlap) + time_scale * ratio_overlap))] = np.true_divide(np.add(
                newSpectrogram[:, int(i * time_scale * (1 - ratio_overlap)):int(
                    i * time_scale * (1 - ratio_overlap) + time_scale * ratio_overlap)],
                toInput[:, :int(time_scale * ratio_overlap)]), 2)
            newSpectrogram[:, int(i * time_scale * (1 - ratio_overlap) + time_scale * ratio_overlap):int(
                i * time_scale * (1 - ratio_overlap) + time_scale)] = toInput[:,
                                                                      -int(time_scale * (1 - ratio_overlap)):]
    return newSpectrogram


def spectrogramToAudioFile(spectrogram, phase=None, fftWindowSize=1024, phaseIterations=10):
    if phase is not None:
        squaredAmplitudeAndSquaredPhase = np.power(spectrogram, 2)
        squaredPhase = np.power(phase, 2)
        unexpd = np.sqrt(np.max(squaredAmplitudeAndSquaredPhase - squaredPhase, 0))
        amplitude = np.expm1(unexpd)
        stftMatrix = amplitude + phase * 1j
        audio = librosa.istft(stftMatrix)
    else:
        amplitude = np.exp(spectrogram) - 1  # 对应log1p
        for i in range(phaseIterations):
            if i == 0:
                reconstruction = np.random.random_sample(amplitude.shape) + 1j * (
                        2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi)
            else:
                reconstruction = librosa.stft(audio, fftWindowSize)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio


def saveSpectrogram(spectrogram, filePath):
    spectrum = spectrogram
    image = np.clip((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)), 0, 1)
    # Low-contrast image warnings are not helpful, tyvm
    io.imsave(filePath, image)


def saveAudioFile(audioFile, filePath, sampleRate=22050):
    librosa.output.write_wav(filePath, audioFile, sampleRate, norm=True)
