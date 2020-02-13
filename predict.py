
import console
import conversion
import os
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate



def isolateVocals(self, path, fftWindowSize, phaseIterations=10):
    # audio stft 得幅值
    console.log("Attempting to isolate vocals from", path)
    audio, sampleRate = conversion.loadAudioFile(path)   # 音频的信号值float ndarray
    spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize=fftWindowSize)     # stft得到的矩阵的幅值和相位
    console.log("Retrieved spectrogram; processing...")

    # 幅值 进模型 得幅值
    # newSpectrogram = self.model.predict(conversion.expandToGrid(spectrogram, self.peakDownscaleFactor)[np.newaxis, :, :, np.newaxis])[0][:spectrogram.shape[0], :spectrogram.shape[1]]
    expandedSpectrogram = conversion.expandToGrid(spectrogram, self.peakDownscaleFactor)   # 幅值放大一些by加0，为了能整除peakDownscaleFactor
    expandedSpectrogramWithBatchAndChannels = expandedSpectrogram[np.newaxis, :, :, np.newaxis]  # 第一维加batch，第四维加channel
    predictedSpectrogramWithBatchAndChannels = self.model.predict(expandedSpectrogramWithBatchAndChannels)   # 放入model做预测
    predictedSpectrogram = predictedSpectrogramWithBatchAndChannels[0, :, :, 0] # o /// o   # 取预测结果的第0batch，第0channel,即再变回幅值
    newSpectrogram = predictedSpectrogram[:spectrogram.shape[0], :spectrogram.shape[1]]    # 幅值缩小为原来的尺寸
    console.log("Processed spectrogram; reconverting to audio")

    # 幅值反stft转为阿卡贝拉audio数
    newAudio = conversion.spectrogramToAudioFile(newSpectrogram, fftWindowSize=fftWindowSize, phaseIterations=phaseIterations)   # phase?
    pathParts = os.path.split(path)
    fileNameParts = os.path.splitext(pathParts[1])
    outputFileNameBase = os.path.join(pathParts[0], fileNameParts[0] + "_acapella")
    console.log("Converted to audio; writing to", outputFileNameBase)

    # 数转音，存
    conversion.saveAudioFile(newAudio, outputFileNameBase + ".wav", sampleRate)
    conversion.saveSpectrogram(newSpectrogram, outputFileNameBase + ".png")
    conversion.saveSpectrogram(spectrogram, os.path.join(pathParts[0], fileNameParts[0]) + ".png")
    console.log("Vocal isolation complete 👌")

def trainModel(epochs=6, batch=8):
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
    weightPath = 'weights' + ".h5"
    model.save_weights(weightPath, overwrite=True)


def train(x, y, trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])


def valid(x, y, trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])


console.log("Weights provided; performing inference on " + str(args.files) + "...")
console.h1("Loading weights")
acapellabot.loadWeights(args.weights)
for f in args.files:
    acapellabot.isolateVocals(f, args.fft, args.phase)
