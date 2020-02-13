import numpy as np

import os.path
import conversion
import data
import os

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
#x = np.array(x)[:, :, :, np.newaxis]
#y = np.array(y)[:, :, :, np.newaxis]
hidden_size = 32

X_ = tf.placeholder(tf.float32, shape=[None, 128, 128], name="input")
Y_ = tf.placeholder(tf.float32, shape=[None, 128, 128], name="true_answer")


# X = tf.reshape(X_, shape=(-1, 128*128))
# Y = tf.reshape(Y_, shape=(-1, 128*128, 1))

def lstm_cell_fw():  # 前向的lstm网络
    return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)


def lstm_cell_bw():  # 反向的lstm网络
    return tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)


stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell_fw() for _ in range(5)])
stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell_bw() for _ in range(5)])

# 输出
outputs, _ = tf.nn.bidirectional_dynamic_rnn(stacked_lstm_fw, stacked_lstm_bw, X_, dtype=tf.float32, time_major=False)
print('outputs0: ', outputs[0])
print('outputs1: ', outputs[1])
out = tf.concat(outputs, 2)

y_pred = tf.layers.dense(inputs=tf.reshape(out, shape=(-1, 128 * hidden_size * 2)), units=128 * 128,
                         activation=tf.nn.relu)

# logits=tf.reshape(Y_, shape=(-1, 128*128)) * tf.log(y_pred)
# loss=-tf.reduce_mean(tf.reduce_sum(logits, axis=1))
loss = 0.5 * tf.reduce_mean(tf.multiply(tf.subtract(tf.reshape(Y_, shape=(-1, 128 * 128)), y_pred),
                                        tf.subtract(tf.reshape(Y_, shape=(-1, 128 * 128)), y_pred)))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())


def train(trainingSplit=0.9):
    return (x[:int(len(x) * trainingSplit)], y[:int(len(y) * trainingSplit)])


def valid(trainingSplit=0.9):
    return (x[int(len(x) * trainingSplit):], y[int(len(y) * trainingSplit):])


print('begin run train()')
xTrain, yTrain = train()
print('begin run valid()')
xValid, yValid = valid()
print('finish run valid()')
print('type(xTrain): ', type(xTrain))


def testing_func(testing_data, testing_label, sess):
    loss_ = sess.run([loss], feed_dict={X_: testing_data, Y_: testing_label})

    print('loss: ', loss_)


def predict_func(testing_data, sess):
    y_pred_ = sess.run([y_pred], feed_dict={X_: testing_data})

    return y_pred_[0]


print()
print("#######################################")
print("training集: ")
testing_func(xTrain, yTrain, sess)
print("#######################################")

print("validation集: ")
testing_func(xValid, yValid, sess)
print("#######################################")

print("开始训练")
saver = tf.train.Saver(max_to_keep=1)
min_loss = 9999
for i in range(2):
    print()
    print("训练轮数: ", i)
    _, loss_ = sess.run([train_op, loss], feed_dict={X_: xTrain, Y_: yTrain})

    if loss_ < min_loss:
        min_loss = loss_
        saver.save(sess, "save2/My_Model.ckpt", global_step=i + 1)

    print("#######################################")
    print("testing: ")
    print("training集: ")
    testing_func(xTrain, yTrain, sess)
    print("#######################################")

    print("validation集: ")
    testing_func(xValid, yValid, sess)
    print("#######################################")

print("Predict:")
sess1 = tf.Session()

model_file = tf.train.latest_checkpoint("save2/")
saver.restore(sess1, model_file)

lstm_y_pred = predict_func(xValid, sess1)
print('y_pred: ')
print(lstm_y_pred)
print(np.array(lstm_y_pred).shape)

np.savetxt('y_pred.txt', lstm_y_pred, fmt='%f', delimiter=' ')

print("#######################################")
