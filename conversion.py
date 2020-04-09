import librosa
import numpy as np
import scipy
import warnings
import skimage.io as io
from os.path import basename
from math import ceil
import argparse
import console
from pydub import AudioSegment
import wave
import struct
import soundfile as sf


def loadAudioFile(filePath,sr=None):
    print('loading audio from:',filePath)
    wf = wave.open(filePath, 'rb')
    nframes = wf.getparams()
    print("saving audio to:", filePath)
    print(nframes)
    wf.close()
    audio, sampleRate = librosa.load(filePath,sr)  # Load an audio file as a floating point time series， 默认采样率sr=44100
    return audio, sampleRate

def saveAudioFile(audioFile, filePath, sampleRate):
    audioFile = np.vstack((audioFile, audioFile)).T
    sf.write(filePath, audioFile, sampleRate)
    wf = wave.open(filePath, 'rb')
    nframes = wf.getparams()
    print("saving audio to:",filePath)
    print(nframes)
    wf.close()


# Return a 2d numpy array of the spectrogram
def audioFileToSpectrogram(audioFile, fftWindowSize):
    spectrogram = librosa.stft(audioFile, fftWindowSize)
    #print("werwerwerwer",spectrogram)# 返回复数值矩阵D , STFT矩阵D中的行数是（1 + 第二个参数一般2的幂n_fft / 2）
    phase = np.imag(spectrogram)       # 为什么不是np.angle？
    amplitude = np.log1p(np.abs(spectrogram))   # np.abs(D[f, t]) is the magnitude of frequency bin f at frame帧 t， loge(1+ )
    return amplitude, phase


import tensorflow as tf

from tensorflow.contrib.signal import stft, hann_window
# pylint: enable=import-error

__email__ = 'research@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'


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





def loadSpectrogram(filePath):
    fileName = basename(filePath)
    if filePath.index("sampleRate") < 0:
        console.warn("Sample rate should be specified in file name", filePath)
        sampleRate = 44100
    else:
        sampleRate = int(fileName[fileName.index("sampleRate=") + 11:fileName.index(").png")])
    console.info("Using sample rate : " + str(sampleRate))
    image = io.imread(filePath, as_grey=True)
    return image / np.max(image), sampleRate

def saveSpectrogram(spectrogram, filePath):
    spectrum = spectrogram
    console.info("Range of spectrum is " + str(np.min(spectrum)) + " -> " + str(np.max(spectrum)))
    image = np.clip((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum)), 0, 1)
    console.info("Shape of spectrum is", image.shape)
    # Low-contrast image warnings are not helpful, tyvm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(filePath, image)
    console.log("Saved image to", filePath)

def fileSuffix(title, **kwargs):
    return " (" + title + "".join(sorted([", " + i + "=" + str(kwargs[i]) for i in kwargs])) + ")"

def handleAudio(filePath, args):
    console.h1("Creating Spectrogram")
    INPUT_FILE = filePath
    INPUT_FILENAME = basename(INPUT_FILE)

    console.info("Attempting to read from " + INPUT_FILE)
    audio, sampleRate = loadAudioFile(INPUT_FILE)
    console.info("Max of audio file is " + str(np.max(audio)))
    spectrogram, phase = audioFileToSpectrogram(audio, fftWindowSize=args.fft)
    SPECTROGRAM_FILENAME = INPUT_FILENAME + fileSuffix("Input Spectrogram", fft=args.fft, iter=args.iter, sampleRate=sampleRate) + ".png"

    saveSpectrogram(spectrogram, SPECTROGRAM_FILENAME)

    print()
    console.wait("Saved Spectrogram; press Enter to continue...")
    print()

    handleImage(SPECTROGRAM_FILENAME, args, phase)

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

def handleImage(fileName, args, phase):
    console.h1("Reconstructing Audio from Spectrogram")

    spectrogram, sampleRate = loadSpectrogram(fileName)
    audio = spectrogramToAudioFile(spectrogram, phase, fftWindowSize=args.fft, phaseIterations=args.iter)

    sanityCheck, phase = audioFileToSpectrogram(audio, fftWindowSize=args.fft)
    saveSpectrogram(sanityCheck, fileName + fileSuffix("Output Spectrogram", fft=args.fft, iter=args.iter, sampleRate=sampleRate) + ".png")

    saveAudioFile(audio, fileName + fileSuffix("Output", fft=args.fft, iter=args.iter) + ".wav", sampleRate)

if __name__ == "__main__":
    # Test code for experimenting with modifying acapellas in image processors (and generally testing the reconstruction pipeline)
    parser = argparse.ArgumentParser(description="Convert image files to audio and audio files to images")
    parser.add_argument("--fft", default=1536, type=int, help="Size of FFT windows")
    parser.add_argument("--iter", default=10, type=int, help="Number of iterations to use for phase reconstruction")
    parser.add_argument("files", nargs="*", default=[])

    args = parser.parse_args()

    for f in args.files:
        if (f.endswith(".mp3") or f.endswith(".wav")):
            handleAudio(f, args)
        elif (f.endswith(".png")):
            handleImage(f, args)
