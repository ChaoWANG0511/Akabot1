import argparse
import os
from tensorflow.keras.metrics import Accuracy
# from data_processing import traversalDir_FirstDir, loadAudioFile, audioFileToSpectrogram, expandToGrid, chop, ichop, \
#     spectrogramToAudioFile, saveAudioFile, saveSpectrogram, song2spectrogram, song2spectrogram_unet, expandToGrid_unet,\
#     ichop_unet, chop_unet
from Data import chop, dataset, ichop, estimateSpectro
from data_processing import saveAudioFile
from models import blstm, apply_unet, apply_blstm, Nest_Net
from tensorflow.keras.models import load_model
from original_aka import console
import numpy as np
import copy
import musdb
from IPython.display import Audio, display
from scipy.signal import stft, istft


class Acapella:
    def __init__(self, weight_path=None):
        self.model = apply_blstm()
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=[Accuracy()])
        self.model_unet = apply_unet()
        self.model_unet.compile(loss='mean_squared_error', optimizer='adam', metrics=[Accuracy()])
        self.model_unetpp = Nest_Net()
        self.model_unetpp.compile(loss='mean_squared_error', optimizer='adam', metrics=[Accuracy()])


    def saveWeights(self, path):
        self.model.save(path+'.h5', overwrite=True)
        self.model_unet.save(path+'_unet.h5',overwrite=True)
        self.model_unetpp.save(path+'_unetpp.h5',overwrite=True)

    def loadWeights(self, path):
        self.model = load_model(path+'.h5')
        self.model_unet = load_model(path+'_unet.h5')
        self.model_unetpp = load_model(path+'_unetpp.h5')

    def train(self, m_path, s_path, instrument):
        mix_path = traversalDir_FirstDir(m_path)
        sou_path = traversalDir_FirstDir(s_path)
        all_path = mix_path.copy()
        for key in all_path.keys():
            all_path[key].extend(sou_path[key])
        print("%d mixtures in the given path to train" % len(mix_path))
        print(all_path)

        left_path_blst = copy.deepcopy(all_path)
        while left_path_blst:
            list_dataset, left_path_blst = song2spectrogram(left_path_blst)#所需内存过大
            if instrument == 'vocals':
                self.model.fit(list_dataset[0], list_dataset[5], epochs=10, batch_size=20, validation_split=0.1)

                self.model.save("vocalweights.h5", overwrite=True)

        left_path_unet = all_path
        while left_path_unet:
            list_dataset_unet, left_path_unet = song2spectrogram_unet(left_path_unet)
            if instrument == 'vocals':
                x_unet = np.array(list_dataset_unet[0])[:, :, :, np.newaxis]
                y_unet = np.array(list_dataset_unet[5])[:, :, :, np.newaxis]
                self.model_unet.fit(x_unet, y_unet, batch_size=50, epochs=1, validation_split=0.1)

                self.model_unet.save("vocalweights_unet.h5", overwrite=True)

            elif instrument == 'bass':
                self.model.fit(list_dataset[0], list_dataset[2], epochs=10, batch_size=20, validation_split=0.1)
            elif instrument == 'drums':
                self.model.fit(list_dataset[0], list_dataset[3], epochs=10, batch_size=20, validation_split=0.1)
            elif instrument == 'other':
                self.model.fit(list_dataset[0], list_dataset[4], epochs=10, batch_size=20, validation_split=0.1)

        save = input("Should we save intermediate weights [y/n]? ")
        if not save.lower().startswith("n"):
            # weightPath = ''.join(random.choice(string.digits) for _ in range(16)) + ".h5"
            weightPath = instrument + 'weights'
            console.log("Saving intermediate weights to", weightPath)
            self.saveWeights(weightPath)

        return self.model

    def train_musdb(self, data):
        smooth = 1.
        dropout_rate = 0.5
        act = "relu"

        X, M = dataset(data)
        print(X.shape, M['vocals'].shape)
        self.model.fit(X[:20,:,:,:], M['vocals'][:20,:,:,:], batch_size=2, epochs=20)
        self.model_unet.fit(X[:20,:,:,:], M['vocals'][:20,:,:,:], batch_size=2, epochs=20)
        self.model_unetpp.fit(X[:20,:,:,:], M['vocals'][:20,:,:,:], batch_size=2, epochs=20)
        self.saveWeights('./model')

        return (self.model, self.model_unet, self.model_unetpp)


    def predict(self, file_path):
        #blstm
        audio, sampleRate = loadAudioFile(file_path)
        spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize=1024)  # stft得到的矩阵的幅值和相位

        expandedSpectrogram, K = expandToGrid(spectrogram, 30, 0.5)
        Slices = chop(expandedSpectrogram, 30, 0.5)  # (1073, 30, 513)
        predictedSlices = self.model.predict(Slices)  # 放入model做预测
        newSpectrogram = ichop(expandedSpectrogram, predictedSlices)

        #unet
        expandedSpectrogram_unet = expandToGrid_unet(spectrogram,64,0.5)
        slices_unet = chop_unet(expandedSpectrogram_unet, 64, 0.5)
        x_test = np.empty((0,512, 64), float)
        x_test = np.append(x_test, slices_unet,axis=0)
        x_test = np.array(x_test)[:, :, :, np.newaxis]
        x_predicted = self.model_unet.predict(x_test)
        x_predicted = np.squeeze(x_predicted, axis = 3)
        newSpectrogram_unet = ichop_unet(x_predicted,64,0.5, expandedSpectrogram_unet)
        b = np.zeros(newSpectrogram_unet.shape[1])
        expandedSpectrogram_unet = np.insert(newSpectrogram_unet, newSpectrogram_unet.shape[0], values=b, axis=0)

        return 1*newSpectrogram+0*expandedSpectrogram_unet[:,0:945]
    
    def predict_musdb(self, track):

        X, M = dataset(track)
        X_origin = stft(track[0].audio.T, nperseg=4096, noverlap=3072)[-1]

        M_predict = self.model.predict(X)
        # M2_predict = self.model_unet.predict(X)
        # M3_predict = self.model_unetpp.predict(X)

        print(M_predict.shape)

        MM_predict = {'vocals': M_predict,
                    'drums': M_predict,
                    'bass': M_predict,
                    'other':M_predict}
        newM = ichop(X_origin, MM_predict)
        estimates = estimateSpectro(X_origin, newM)
        return estimates

        
        

if __name__ == "__main__":
    # if data folder is specified, create a new data object and train on the data
    # if input audio is specified, infer on the input
    parser = argparse.ArgumentParser(description="Acapella extraction with a convolutional neural network")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--data", default=None, type=str, help="Path containing training data")
    parser.add_argument("--split", default=0.9, type=float, help="Proportion of the data to train on")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to train.")
    parser.add_argument("--weights", default="./model", type=str, help="h5 file to read/write weights to")
    parser.add_argument("--batch", default=8, type=int, help="Batch size for training")
    parser.add_argument("--phase", default=10, type=int, help="Phase iterations for reconstruction")
    parser.add_argument("--load", action='store_true', help="Load previous weights file before starting")
    parser.add_argument("--mp3", default='dataset/audio_example.mp3', type=str, help="Path of file will be processed")
    parser.add_argument("command", default='')
    parser.add_argument("--instrument", default='vocals', help="The target instrument")

    args = parser.parse_args()
    acapellabot = Acapella()
    mus = musdb.DB(download=True, subsets='train')

    if args.command == 'train':
        model = acapellabot.train('DSD100subset/Mixtures/Test', 'DSD100subset/Sources/Test', args.instrument)

    if args.command == 'train_musdb':
        (blstm, unet, unetpp) = acapellabot.train_musdb(mus[0:30])

    if args.command == 'predict_musdb':

        track = [mus[1]]

        acapellabot.loadWeights(args.weights)
        result = acapellabot.predict_musdb(track)

        for target, estimate in result.items():
            print(str(target)+str(estimate))
            saveAudioFile(np.asfortranarray(estimate.T), "./result.wav")

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
