import argparse
import os
from tensorflow.keras.metrics import Accuracy
from data_processing import traversalDir_FirstDir, loadAudioFile, audioFileToSpectrogram, expandToGrid, chop, ichop, \
    spectrogramToAudioFile, saveAudioFile, saveSpectrogram, song2spectrogram, song2spectrogram_unet
from models import blstm, apply_unet
from tensorflow.keras.models import load_model
from original_aka import console
import numpy as np


class Acapella:
    def __init__(self, weight_path=None):
        self.model = blstm()
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=[Accuracy()])
        self.model_unet = apply_unet()
        self.model_unet.compile(loss='mean_squared_error', optimizer='adam', metrics=[Accuracy()])

    def saveWeights(self, path):
        self.model.save(path, overwrite=True)
        self.model_unet.save(path+'_unet',overwrite=True)

    def loadWeights(self, path):
        self.model = load_model(path)
        self.model_unet = load_model(path+'_unet')

    def train(self, m_path, s_path, instrument):
        mix_path = traversalDir_FirstDir(m_path)
        sou_path = traversalDir_FirstDir(s_path)
        all_path = mix_path.copy()
        for key in all_path.keys():
            all_path[key].extend(sou_path[key])
        print("%d mixtures in the given path to train" % len(mix_path))
        print(all_path)
        list_dataset = song2spectrogram(all_path)
        print('phase： ', len(list_dataset[1]), list_dataset[1][0].shape)
        print('dataset for model: ', list_dataset[0].shape, list_dataset[3].shape)

        list_dataset_unet = song2spectrogram_unet(all_path_para=all_path)

        if instrument == 'vocals':
            x_unet = np.array(list_dataset_unet[0])[:, :, :, np.newaxis]
            y_unet = np.array(list_dataset_unet[0])[:, :, :, np.newaxis]
            self.model_unet.fit(x_unet, y_unet, batch_size=50, epochs=1, validation_split=0.1)
            self.model.fit(list_dataset[0], list_dataset[5], epochs=10, batch_size=20, validation_split=0.1)
        elif instrument == 'bass':
            self.model.fit(list_dataset[0], list_dataset[2], epochs=10, batch_size=20, validation_split=0.1)
        elif instrument == 'drums':
            self.model.fit(list_dataset[0], list_dataset[3], epochs=10, batch_size=20, validation_split=0.1)
        elif instrument == 'other':
            self.model.fit(list_dataset[0], list_dataset[4], epochs=10, batch_size=20, validation_split=0.1)

        # BLSTM model
        # self.model.fit(list_dataset[0], list_dataset[-1], epochs=10, batch_size=20, validation_split=0.1)
        # self.model.summary()

        save = input("Should we save intermediate weights [y/n]? ")
        if not save.lower().startswith("n"):
            # weightPath = ''.join(random.choice(string.digits) for _ in range(16)) + ".h5"
            weightPath = instrument + 'weights.h5'
            console.log("Saving intermediate weights to", weightPath)
            self.saveWeights(weightPath)

        return self.model

    def predict(self, file_path):
        audio, sampleRate = loadAudioFile(file_path)
        spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize=1024)  # stft得到的矩阵的幅值和相位
        expandedSpectrogram, K = expandToGrid(spectrogram, 30, 0.5)
        Slices = chop(expandedSpectrogram, 30, 0.5)  # (1073, 30, 513)
        predictedSlices = self.model.predict(Slices)  # 放入model做预测
        newSpectrogram = ichop(expandedSpectrogram, predictedSlices)
        return newSpectrogram


if __name__ == "__main__":
    # if data folder is specified, create a new data object and train on the data
    # if input audio is specified, infer on the input
    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--data", default=None, type=str, help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float, help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default="weights.h5", type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--phase", default=10, type=int, help="Phase iterations for reconstruction")
    parser.add_argument("--load", action='store_true', help="Load previous weights file before starting")
    parser.add_argument("--mp3", default='dataset/audio_example.mp3', type=str, help="Path of file will be processed")
    parser.add_argument("command", default='')
    parser.add_argument("--instrument", default='vocals', help="The target instrument")

    args = parser.parse_args()
    acapellabot = Acapella()

    if args.command == 'train':
        model = acapellabot.train('DSD100subset/Mixtures/Test', 'DSD100subset/Sources/Test', args.instrument)

    if args.command == 'predict':
        acapellabot.loadWeights(args.weights)
        newSpectrogram = acapellabot.predict(args.mp3)
        # spectrogram retransform and save
        newAudio = spectrogramToAudioFile(newSpectrogram)  # phase?   ,newphase
        outputFileNameBase = os.path.join('./result')
        saveAudioFile(newAudio, outputFileNameBase + ".wav")
        saveSpectrogram(newSpectrogram, outputFileNameBase + ".png")
        audio, sampleRate = loadAudioFile('dataset/audio_example.mp3')
        spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize=1024)
        saveSpectrogram(spectrogram, './src_vocal.png')
