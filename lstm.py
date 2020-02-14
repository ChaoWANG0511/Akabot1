import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model

#path_list = ['DSD100subset/Mixtures/Dev/055/mixture.wav', 'DSD100subset/Sources/Dev/055/vocals.wav']

import os.path
import conversion
import data
import os

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense

from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Reshape

# 定义一个函数，path为你的路径
def traversalDir_FirstDir(path):
    # path 输入是 Dev/Test
    # 定义一个字典，用来存储结果————歌名：路径
    dict = {}
    # 获取该目录下的所有文件夹目录, 每个歌的文件夹

    files = os.listdir(path)
    for file in files:
        # 得到该文件下所有目录的路径
        m = os.path.join(path, file)
        h = os.path.split(m)
        dict[h[1]] = []
        song_wav = os.listdir(m)
        m = m + '/'
        for track in song_wav:
            value = os.path.join(m, track)
            dict[h[1]].append(value)
    return dict

mix_path = traversalDir_FirstDir('DSD100subset/Mixtures/Dev/')

sou_path = traversalDir_FirstDir('DSD100subset/Sources/Dev/')

all_path = mix_path.copy()
for key in all_path.keys():
    all_path[key].extend(sou_path[key])

print(all_path)


def preprocessing(all_path_para=all_path, fftWindowSize=1536, SLICE_SIZE=128):
    x = []
    y = []

    for key in all_path_para:
        path_list = [all_path[key][0], all_path[key][-1]]

        for path in path_list:
            audio, sampleRate = conversion.loadAudioFile(path)
            spectrogram, phase = conversion.audioFileToSpectrogram(audio, fftWindowSize)
            print(spectrogram.shape)

            # chop into slices so everything's the same size in a batch 切为模型输入尺寸
            dim = SLICE_SIZE
            Slices = data.chop(spectrogram, dim)   # 114个128*128
            print(len(Slices))
            if 'mixture' in path:
                x.extend(Slices)
            else:
                y.extend(Slices)

    return [x,y]


[x,y] = preprocessing(all_path_para=all_path, fftWindowSize=1536, SLICE_SIZE=128)
x = np.asarray(x)
y = np.asarray(y)
print(x.shape)  # samples, time steps, and features.



model = Sequential()
np.squeeze()
input_tensor = Input(shape=(None, None, 1), name='input')


model.add(Bidirectional(LSTM(32, input_shape = (128,128), return_sequences=True)))
model.add(Bidirectional(LSTM(12, return_sequences=True)))
model.add(Bidirectional(LSTM(6, return_sequences=False)))
model.add(Dense(128 * 128))
model.add(Reshape((128,128)))
#model.add(Dense(128, activation='relu'))
#model.add(Conv2D(222,128,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
#print(model.summary())
# train LSTM
def train(trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])

def valid(trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])

xTrain, yTrain = train()
xValid, yValid = valid()

model.fit(xTrain, yTrain, batch_size=30, epochs=10, validation_data=(xValid, yValid))
weightPath = 'C:\\Users\\chaow\\PycharmProjects\\Akabot\\outweight' + ".h5"
model.save_weights(weightPath, overwrite=True)

#model.fit(x, y, epochs=10, batch_size=30, verbose=2)


conv7 = conv2d_factory(1, (5, 5))(_)

co7resh = squeeze(conv7, -1)
print(co7resh)

shape_unet_bottom = conv6.shape

ls1 = Bidirectional(LSTM(32, return_sequences=True))(co7resh)
ls2 = Bidirectional(LSTM(12, return_sequences=True))(ls1)
ls3 = Bidirectional(LSTM(6, return_sequences=False))(ls2)
ls4 = Dense(shape_unet_bottom[1] * shape_unet_bottom[2])(ls3)
ls5 = Reshape(shape_unet_bottom)(ls4)





