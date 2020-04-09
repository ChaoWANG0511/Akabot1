import librosa
import tensorflow as tf
from tensorflow.contrib.signal import stft, hann_window
import os
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model
from math import ceil
import argparse
import random
import string
import os
import numpy as np
import sys
sys.path.append("..")
import conversion
from data import Data
from unet import apply_unet
import museval


def spectrogramToAudioFile(spectrogram, phase, fftWindowSize, phaseIterations=10):
    if phase is not None:
        squaredAmplitudeAndSquaredPhase = np.power(spectrogram, 2)
        squaredPhase = np.power(phase, 2)
        unexpd = np.sqrt(np.max(squaredAmplitudeAndSquaredPhase - squaredPhase, 0))
        amplitude = np.expm1(unexpd)
        stftMatrix = amplitude + phase * 1j
        audio = librosa.istft(stftMatrix)
    else:
        amplitude = np.exp(spectrogram) - 1   # 对应log1p
        for i in range(phaseIterations):
            if i == 0:
                reconstruction = np.random.random_sample(amplitude.shape) + 1j * (2 * np.pi * np.random.random_sample(amplitude.shape) - np.pi)
            else:
                reconstruction = librosa.stft(audio, fftWindowSize)
            spectrum = amplitude * np.exp(1j * np.angle(reconstruction))
            audio = librosa.istft(spectrum)
    return audio


def chop(matrix, time_scale, ratio_overlap):
    slices = []
    for time in range(0, ceil((matrix.shape[1]-time_scale)/time_scale/(1-ratio_overlap))+1):   # 列
            s = matrix[: (int(matrix.shape[0]/64))*64,
                       int(time * time_scale*(1-ratio_overlap)) :int(time * time_scale*(1-ratio_overlap))+time_scale ]
            print(s.shape)
            slices.append(s)
    return slices

def expandToGrid(spectrogram, time_scale, ratio_overlap):
    newY = int(ceil((spectrogram.shape[1] - time_scale)/(1-ratio_overlap)/time_scale) * (1-ratio_overlap)*time_scale)+time_scale  # ceil取上
    newX = int(spectrogram.shape[0]/64)*64
    newSpectrogram = np.zeros((newX, newY))   # 右，下：多一些0，至能被girdsize整除
    print(spectrogram.shape,"original shape")
    print(newSpectrogram.shape,"expanded shape")
    newSpectrogram[:spectrogram.shape[0], :spectrogram.shape[1]] = spectrogram[:int(spectrogram.shape[0]/64)*64,:]
    return newSpectrogram


def ichop(predicted_slices, time_scale, ratio_overlap,phase_to_fit):
    cutpoint=int(time_scale*ratio_overlap)
    cutpointB=time_scale-cutpoint
    newspectrogram=np.zeros((predicted_slices[0].shape[0],cutpointB))
    for i in range(len(predicted_slices)):
        front=predicted_slices[i][:,:cutpoint]
        middle=predicted_slices[i][:,cutpoint:cutpointB]
        back=predicted_slices[i-1][:,cutpointB:]
        if i==0:
            newspectrogram=np.hstack((front,middle))
        elif i==len(predicted_slices)-1:
            newspectrogram=np.hstack((newspectrogram,0.5*(front+back),middle,predicted_slices[i][:,cutpointB:]))
        else:
            newspectrogram=np.hstack((newspectrogram, 0.5*(front+back),middle))
    newspectrogram_fit=newspectrogram[:phase_to_fit.shape[0],:phase_to_fit.shape[1]]
    return newspectrogram_fit


class AcapellaBot:
    # 定义model
    def __init__(self):

        m = apply_unet()

        m.compile(loss='mean_squared_error', optimizer='adam')

        self.model = m
        self.peakDownscaleFactor = 4

    # 训练验证model.fit, 给data, epochs, batch, 是否存权重
    def train(self, data, epochs, batch=1):
        xTrain, yTrain = data.train()
        xValid, yValid = data.valid()


        self.model.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))


        weightPath = ''.join(random.choice(string.digits) for _ in range(16)) + ".h5"
        self.saveWeights(weightPath)


    def saveWeights(self, path):
        self.model.save_weights(path, overwrite=True)
    def loadWeights(self, path):
        self.model.load_weights(path)

    # 预测model， 数据处理stft
    def isolateVocals(self, path, fftWindowSize, phaseIterations=10,time_scale=256,ratio_overlap=0.2):
        # audio stft 得幅值
        audio, sampleRate = conversion.loadAudioFile(path)   # 音频的信号值float ndarray
        #print(audio,"yyyyyyyyyyyyy")
        spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize=fftWindowSize)     # stft得到的矩阵的幅值和相位
        #print("ooooooooooooooooooooooooooooooo",spectrogram)

        expandedSpectrogram = expandToGrid(spectrogram,time_scale,ratio_overlap)
        newphase=expandToGrid(phase,time_scale,ratio_overlap)
        Slices = chop(expandedSpectrogram, time_scale, ratio_overlap)
#        print(Slices.shape)
        predicted_slices=[]
        for slice in Slices:

            expandedsliceWithBatchAndChannels = slice[np.newaxis, :, :, np.newaxis]  # 第一维加batch，第四维加channel
            predictedsliceWithBatchAndChannels = self.model.predict(expandedsliceWithBatchAndChannels)   # 放入model做预测
            predictedslicewithzeros = predictedsliceWithBatchAndChannels[0, :, :, 0] # o /// o   # 取预测结果的第0batch，第0channel,即再变回幅值
            predicted_slice = predictedslicewithzeros[:slice.shape[0], :slice.shape[1]]    # 幅值缩小为原来的尺寸
            predicted_slices.append(predicted_slice)

        newSpectrogram=ichop(predicted_slices,time_scale,ratio_overlap,newphase)
        #print(newSpectrogram,"ddddddddddddc")
        #print(newSpectrogram.shape,"ichopped shape")


        # 幅值反stft转为阿卡贝拉audio数
        newAudio = spectrogramToAudioFile(newSpectrogram,newphase, fftWindowSize=fftWindowSize, phaseIterations=phaseIterations)   # phase?
        pathParts = os.path.split(path)
        fileNameParts = os.path.splitext(pathParts[1])
        outputFileNameBase = os.path.join(pathParts[0], "vocals")
        eval_est_path=  os.path.join(pathParts[0],"eval")
        eval_rslt_path=os.path.splitext(pathParts[0])
        rslt_path=''.join(eval_rslt_path)

        if not os.path.exists(eval_est_path):
            os.mkdir(eval_est_path)


        # 数转音，存
        #print(newAudio,"qqqqqqqqqqqqqqq")
        conversion.saveAudioFile(newAudio, outputFileNameBase + ".wav", sampleRate)# should be .../mixture_acapella.wav
        conversion.saveAudioFile(newAudio, os.path.join(pathParts[0],"eval","vocals")+".wav", sampleRate)# should be .../mixture_acapella.wav


        conversion.saveSpectrogram(newSpectrogram, outputFileNameBase + ".png")# should ba .../mixture_acapella.png
        conversion.saveSpectrogram(expandedSpectrogram, os.path.join(pathParts[0], fileNameParts[0]) + ".png")# should be .../mixture.png


        #evaluation
        path=path.replace("Mixtures","Sources")
        path=path.replace("mixture","vocals")# path for acapella reference for this mixture

        audio, sampleRate = conversion.loadAudioFile(path)

        spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize=fftWindowSize)

        expandedTargetgram = expandToGrid(spectrogram, time_scale, ratio_overlap)
        newphase=expandToGrid(phase,time_scale,ratio_overlap)

        expandedTargetAudio = spectrogramToAudioFile(expandedTargetgram, newphase, fftWindowSize=fftWindowSize,
                                          phaseIterations=phaseIterations)

        conversion.saveSpectrogram(expandedTargetgram, os.path.join(pathParts[0], fileNameParts[0]+"_target.png"))# save vocal reference spectrogram to mixture source path
        #conversion.saveAudioFile(expandedTargetAudio, path, sampleRate)# save vocal ref audio to from where it came

        pathParts = os.path.split(path)
        eval_ref_path = os.path.join(pathParts[0],"eval")
        if not os.path.exists(eval_ref_path):
            os.mkdir(eval_ref_path)

        conversion.saveAudioFile(expandedTargetAudio, os.path.join(pathParts[0],"eval/vocals.wav"), sampleRate)# save vocal ref audio to a new sub folder under from where it came

        path_ref=''.join(eval_ref_path)
        print("reference:",path_ref)
        path_est=''.join(eval_est_path)
        print("estimate:",path_est)
        a = museval.eval_dir(path_ref, path_est,output_dir=rslt_path)
        print(a)





if __name__ == "__main__":
    # if data folder is specified, create a new data object and train on the data
    # if input audio is specified, infer on the input
    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--data", default=None, type=str, help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float, help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default="../weights_kkk.h5", type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=1, type=int, help="Batch size for training")
    parser.add_argument("--phase", default=10, type=int, help="Phase iterations for reconstruction")
    parser.add_argument("--load", action='store_true', help="Load previous weights file before starting")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    acapellabot = AcapellaBot()

    # 给data: 训练
    if len(args.files) == 0 and args.data:
        if args.load:
            acapellabot.loadWeights(args.weights)
        data = Data(args.data, args.fft, args.split)   # 训练用的data见data.py中的类Data
        acapellabot.train(data, args.epochs, args.batch)
        acapellabot.saveWeights(args.weights)

    # 给files: 预测
    elif len(args.files) > 0:
        acapellabot.loadWeights(args.weights)
        for f in args.files:
            acapellabot.isolateVocals(f,args.fft)

    else:
        print("Please provide data to train on (--data) or files to infer on")
