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

import tensorflow as tf

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
print(y.shape)


hidden_size = 16


X_ = tf.placeholder(tf.float32, shape=[None, 128, 128], name="input")
Y = tf.placeholder(tf.float32, shape=[None, 128, 128], name="true_answer")
#X = tf.reshape(X_, shape=[-1, 128*128])

def lstm_cell_fw():     # 前向的lstm网络
    return tf.contrib.rnn.BasicLSTMCell(hidden_size,forget_bias=1.0, state_is_tuple=True)
def lstm_cell_bw():     # 反向的lstm网络
    return tf.contrib.rnn.BasicLSTMCell(hidden_size,forget_bias=1.0, state_is_tuple=True)

stacked_lstm_fw=tf.contrib.rnn.MultiRNNCell([lstm_cell_fw() for _ in range(3)])
stacked_lstm_bw=tf.contrib.rnn.MultiRNNCell([lstm_cell_bw() for _ in range(3)])

# 输出
outputs, _ = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw, inputs = X_, dtype=tf.float32, time_major=False)
out = tf.concat(outputs, 2)
y_pred = tf.layers.dense(inputs=tf.reshape(out, shape=(-1, 128*hidden_size*2)), units=128*128, activation=tf.nn.relu);

# loss function
logits=tf.reshape(Y, shape=(-1, 128*128)) * tf.log(y_pred)
loss=-tf.reduce_mean(tf.reduce_sum(logits, axis=1))
#loss = 0.5* tf.reduce_sum(tf.subtract(tf.reshape(Y, shape=(-1, 128*128)), y_pred)) * tf.reduce_sum(tf.subtract(tf.reshape(Y, shape=(-1, 128*128)), y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

sess = tf.Session();
sess.run(tf.initialize_all_variables())

def train(trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])

def valid(trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])

xTrain, yTrain = train()
xValid, yValid = valid()

def testing_func(testing_data, testing_label, sess):
    loss_ = sess.run([loss], feed_dict={X_: testing_data, Y: testing_label})
    print('loss: ', loss_)


print()
print("开始训练")
for i in range(20):
    print()
    print("训练轮数: ", i)
    _, loss_ = sess.run([train_op, loss], feed_dict={X_: xTrain, Y: yTrain})
        
    print("#######################################")
    print("testing: ")
    print("training集: ")
    testing_func(xTrain, yTrain, sess)
    print("#######################################")
    
    print("validation集: ")
    testing_func(xValid, yValid, sess)
    print("#######################################")









