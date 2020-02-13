import numpy as np
import librosa
import tensorflow as tf
from tensorflow.contrib.signal import stft, hann_window
import os
import conversion
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model
from math import ceil
'''

def audioFileToSpectrogram(audioFile, fftWindowSize):
    spectrogram = librosa.stft(audioFile, fftWindowSize)   # 返回复数值矩阵D , STFT矩阵D中的行数是（1 + 第二个参数一般2的幂n_fft / 2）
    phase = np.imag(spectrogram)       # 为什么不是np.angle？
    amplitude = np.log1p(np.abs(spectrogram))   # np.abs(D[f, t]) is the magnitude of frequency bin f at frame帧 t， loge(1+ )
    return amplitude, phase

def compute_spectrogram_tf(
        waveform,
        frame_length=2048, frame_step=512,
        spec_exponent=1., window_exponent=1.):
    """ Compute magnitude / power spectrogram from waveform as
    a n_samples x n_channels tensor.

    :param waveform:        Input waveform as (times x number of channels)
                            tensor.
    :param frame_length:    Length of a STFT frame to use.
    :param frame_step:      HOP between successive frames.
    :param spec_exponent:   Exponent of the spectrogram (usually 1 for
                            magnitude spectrogram, or 2 for power spectrogram).
    :param window_exponent: Exponent applied to the Hann windowing function
                            (may be useful for making perfect STFT/iSTFT
                            reconstruction).
    :returns:   Computed magnitude / power spectrogram as a
                (T x F x n_channels) tensor.
    """
    stft_tensor = tf.transpose(
        stft(
            tf.transpose(waveform),
            frame_length,
            frame_step,
            window_fn=lambda f, dtype: hann_window(
                f,
                periodic=True,
                dtype=waveform.dtype) ** window_exponent),
        perm=[1, 2, 0])
    return np.abs(stft_tensor) ** spec_exponent

def spectrogramToAudioFile(spectrogram, phase, fftWindowSize, phaseIterations=10):
    if phase is not None:
        # reconstructing the new complex matrix
        squaredAmplitudeAndSquaredPhase = np.power(spectrogram, 2)
        squaredPhase = np.power(phase, 2)
        unexpd = np.sqrt(np.max(squaredAmplitudeAndSquaredPhase - squaredPhase, 0))
        amplitude = np.expm1(unexpd)
        stftMatrix = amplitude + phase * 1j
        audio = librosa.istft(stftMatrix)
    else:
        # phase reconstruction with successive approximation
        # credit to https://dsp.stackexchange.com/questions/3406/reconstruction-of-audio-signal-from-its-absolute-spectrogram/3410#3410
        # for the algorithm used
        amplitude = np.exp(spectrogram) - 1   # 对应log1p
        for i in range(phaseIterations):
            if i == 0:
                reconstruction = np.random.random_sample(amplitude.shape) + 1j * (2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi)
            else:
                reconstruction = librosa.stft(audio, fftWindowSize)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio

'''


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
    weightPath = 'weights_cus' + ".h5"
    model.save_weights(weightPath, overwrite=True)


trainModel()
