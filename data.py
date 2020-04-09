"""
Loads and stores mashup data given a folder full of acapellas and instrumentals
Assumes that all audio clips (wav, mp3) in the folder
a) have their Camelot key as the first token in the filename
b) are in the same BPM
c) have "acapella" somewhere in the filename if they're an acapella, and are otherwise instrumental
d) all have identical arrangements
e) have the same sample rate
给定一个充满acapellas和instrumentals的文件夹，加载并存储mashup数据
假定文件夹中的所有音频片段（wav，mp3）都：
a）将其Camelot键作为文件名中的第一个标记
b）在同一BPM中
c）如果是acapella，则在文件名中的某个位置包含“ acapella”，否则是instrumental
d）arrangements相同
e）具有相同的采样率
"""
import sys
import os
import numpy as np
import h5py

import console
import conversion

def listdirInMac(path):
    os_list = os.listdir(path)
    for item in os_list:
        if item.startswith('.') and os.path.isfile(os.path.join(path, item)):
            os_list.remove(item)
    return os_list

def traversalDir_FirstDir(path):
    # path 输入是 Dev/Test
    # 定义一个字典，用来存储结果————歌名：路径
    dict = {}
    # 获取该目录下的所有文件夹目录, 每个歌的文件夹

    files = listdirInMac(path)
    for file in files:
        # 得到该文件下所有目录的路径
        m = os.path.join(path, file)
        h = os.path.split(m)
        dict[h[1]] = []
        song_wav = listdirInMac(m)
        m = m + '/'
        for track in song_wav:
            value = os.path.join(m, track)
            dict[h[1]].append(value)
    return dict


# Modify these functions if your data is in a different format
def keyOfFile(fileName):
    firstToken = int(fileName.split()[0])   # camelot key as key：B小调
    if 0 < firstToken <= NUMBER_OF_KEYS:
        return firstToken
    console.warn("File", fileName, "doesn't specify its key, ignoring..")
    return None


def fileIsAcapella(fileName):
    return "acapella" in fileName.lower()    #acapella在文件名里


NUMBER_OF_KEYS = 12 # number of keys to iterate over
SLICE_SIZE = 128    # size of spectrogram slices to use

# Slice up matrices into squares so the neural net gets a consistent size for training (doesn't matter for inference)
def chop(matrix, scale):
    slices = []
    for time in range(0, matrix.shape[1] // scale):   # 列
        for freq in range(0, matrix.shape[0] // scale):  # 行
            s = matrix[freq * scale : (freq + 1) * scale,
                       time * scale : (time + 1) * scale]
            slices.append(s)  # 切为很多小块，每块scale*scale，存到slices列表， 使得模型的输入都一样大
    return slices


class Data:

    def __init__(self, inPath, fftWindowSize=1536, trainingSplit=0.9):
        self.inPath = inPath
        self.fftWindowSize = fftWindowSize
        self.trainingSplit = trainingSplit
        self.x = []
        self.y = []
        self.peakDownscaleFactor = 4
        self.load_customized()

    # x,y 按trainingSplit比例分训练集验证集
    def train(self):
        return (self.x[:int(len(self.x) * self.trainingSplit)], self.y[:int(len(self.y) * self.trainingSplit)])
    def valid(self):
        return (self.x[int(len(self.x) * self.trainingSplit):], self.y[int(len(self.y) * self.trainingSplit):])

    def load(self, saveDataAsH5=False):
        h5Path = os.path.join(self.inPath, "data.h5")
        if os.path.isfile(h5Path):
            h5f = h5py.File(h5Path, "r")  # 只读h5文件
            self.x = h5f["x"][:]   # 键x,y
            self.y = h5f["y"][:]
        else:
            acapellas = {}
            instrumentals = {}
            # Hash bins for each camelot key so we can merge
            # in the future, this should be a generator w/ yields in order to eat less memory
            for i in range(NUMBER_OF_KEYS):
                key = i + 1
                acapellas[key] = []
                instrumentals[key] = []
            for dirPath, dirNames, fileNames in os.walk(self.inPath):
                for fileName in filter(lambda f : (f.endswith(".mp3") or f.endswith(".wav")) and not f.startswith("."), fileNames):
                    key = keyOfFile(fileName)
                    if key:    #如果有调
                        targetPathMap = acapellas if fileIsAcapella(fileName) else instrumentals     # 根据文件名标tag
                        tag = "[Acapella]" if fileIsAcapella(fileName) else "[Instrumental]"

                        audio, sampleRate,sr,nc = conversion.loadAudioFile(os.path.join(self.inPath, fileName))
                        spectrogram, phase = conversion.audioFileToSpectrogram(audio, self.fftWindowSize)

                        targetPathMap[key].append(spectrogram)                      # ex. acapellas[调]=[这首歌的频谱幅值,...]
                        console.info(tag, "Created spectrogram for", fileName, "in key", key, "with shape", spectrogram.shape)

            # Merge mashups
            for k in range(NUMBER_OF_KEYS):  # 循环每个调：
                acapellasInKey = acapellas[k + 1]
                instrumentalsInKey = instrumentals[k + 1]
                count = 0
                for acapella in acapellasInKey:   # 循环每个阿卡：
                    for instrumental in instrumentalsInKey:      # 循环每个伴奏
                        # Pad if smaller   如果伴奏和阿卡不一样大，则补到一样大
                        if (instrumental.shape[1] < acapella.shape[1]):
                            newInstrumental = np.zeros(acapella.shape)
                            newInstrumental[:instrumental.shape[0], :instrumental.shape[1]] = instrumental
                            instrumental = newInstrumental
                        elif (acapella.shape[1] < instrumental.shape[1]):
                            newAcapella = np.zeros(instrumental.shape)
                            newAcapella[:acapella.shape[0], :acapella.shape[1]] = acapella
                            acapella = newAcapella

                        # simulate a limiter/low mixing (loses info, but that's the point)
                        # I've tested this against making the same mashups in Logic and it's pretty close
                        mashup = np.maximum(acapella, instrumental)   # 取max为混音mashup

                        # chop into slices so everything's the same size in a batch 切为模型输入尺寸
                        dim = SLICE_SIZE
                        mashupSlices = chop(mashup, dim)
                        acapellaSlices = chop(acapella, dim)
                        count += 1        #count 作用？

                        #存入x,y
                        self.x.extend(mashupSlices)
                        self.y.extend(acapellaSlices)
                console.info("Created", count, "mashups for key", k, "with", len(self.x), "total slices so far")

            # Add a "channels" channel to please the network
            self.x = np.array(self.x)[:, :, :, np.newaxis]
            self.y = np.array(self.y)[:, :, :, np.newaxis]

            # Save to file if asked
            if saveDataAsH5:
                h5f = h5py.File(h5Path, "w")
                h5f.create_dataset("x", data=self.x)
                h5f.create_dataset("y", data=self.y)
                h5f.close()

    def load_customized(self, saveDataAsH5=False):

        h5Path = os.path.join(self.inPath, "data.h5")
        if os.path.isfile(h5Path):
            h5f = h5py.File(h5Path, "r")  # 只读h5文件
            self.x = h5f["x"][:]  # 键x,y
            self.y = h5f["y"][:]
        else:
            mix_path = traversalDir_FirstDir(self.inPath + '/Mixtures/Dev')
            sou_path = traversalDir_FirstDir(self.inPath + '/Sources/Dev')
            all_path = mix_path.copy()

            for key in all_path.keys():
                all_path[key].extend(sou_path[key])

            for key in all_path:
                path_list = [all_path[key][0], all_path[key][-1]]

                for path in path_list:
                    audio, sampleRate,sr,nc = conversion.loadAudioFile(path)
                    spectrogram, phase = conversion.audioFileToSpectrogram(audio, self.fftWindowSize)

                    # expandedSpectrogram = conversion.expandToGrid(spectrogram, self.peakDownscaleFactor)

                    # chop into slices so everything's the same size in a batch 切为模型输入尺寸
                    dim = SLICE_SIZE
                    Slices = chop(spectrogram, dim)  # 114个128*128
                    if 'mixture' in path:
                        self.x.extend(Slices)
                    else:
                        self.y.extend(Slices)

            self.x = np.array(self.x)[:, :, :, np.newaxis]
            self.y = np.array(self.y)[:, :, :, np.newaxis]

            if saveDataAsH5:
                h5f = h5py.File(h5Path, "w")
                h5f.create_dataset("x", data=self.x)
                h5f.create_dataset("y", data=self.y)
                h5f.close()

if __name__ == "__main__":
    # Simple testing code to use while developing
    console.h1("Loading Data")
    d = Data(sys.argv[1], 1536)
    console.h1("Writing Sample Data")
    conversion.saveSpectrogram(d.x[0], "x_sample_0.png")
    conversion.saveSpectrogram(d.y[0], "y_sample_0.png")
    audio = conversion.spectrogramToAudioFile(d.x[0], 1536)
    conversion.saveAudioFile(audio, "x_sample.wav", 44100,2,2)
