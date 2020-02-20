import numpy as np
import os
import conversion
from keras.layers import Input, Reshape, Lambda
from keras.models import Model
from math import ceil
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    multiply,
    ReLU)
from functools import partial
from keras.initializers import he_uniform
from keras.backend import int_shape, squeeze
from keras.layers.convolutional_recurrent import ConvLSTM2D


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
            newspectrogram=expandToGrid(spectrogram, time_scale, ratio_overlap)
            Slices = chop(newspectrogram, time_scale, ratio_overlap)
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








def apply_unet():

    conv_n_filters = [16, 32, 64, 128, 256, 512]
    conv_activation_layer = ReLU()
    deconv_activation_layer = ReLU()
    kernel_initializer = he_uniform(seed=50)
    conv2d_factory = partial(
        Conv2D,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)

    input_tensor = Input(batch_shape=(1,768,256, 1), name='input')
    print(input_tensor.shape)
    # First layer.
    conv1 = conv2d_factory(conv_n_filters[0], (5, 5))(input_tensor)
    batch1 = BatchNormalization(axis=-1)(conv1)
    rel1 = conv_activation_layer(batch1)
    # Second layer.
    conv2 = conv2d_factory(conv_n_filters[1], (5, 5))(rel1)
    batch2 = BatchNormalization(axis=-1)(conv2)
    rel2 = conv_activation_layer(batch2)
    # Third layer.
    conv3 = conv2d_factory(conv_n_filters[2], (5, 5))(rel2)
    batch3 = BatchNormalization(axis=-1)(conv3)
    rel3 = conv_activation_layer(batch3)
    # Fourth layer.
    conv4 = conv2d_factory(conv_n_filters[3], (5, 5))(rel3)
    batch4 = BatchNormalization(axis=-1)(conv4)
    rel4 = conv_activation_layer(batch4)
    # Fifth layer.
    conv5 = conv2d_factory(conv_n_filters[4], (5, 5))(rel4)
    batch5 = BatchNormalization(axis=-1)(conv5)
    rel5 = conv_activation_layer(batch5)
    # Sixth layer
    conv6 = conv2d_factory(conv_n_filters[5], (5, 5))(rel5)

    print(1,int_shape(conv1))
    print(5,int_shape(conv5))
    print(6,int_shape(conv6))

    conv_to_LSTM_dims = int_shape(conv6)
    x = Reshape(target_shape = conv_to_LSTM_dims, name='reshapeconvtolstm')(conv6)


    x = ConvLSTM2D(filters= 32, kernel_size=(3, 3),
                   padding='same', return_sequences=True, stateful=True)(x)


    x = ConvLSTM2D(filters=conv_n_filters[5], kernel_size=(3, 3),
                   padding='same', return_sequences=True, stateful=True)(x)

    print(int_shape(x))


    LSTM_to_conv_dims = int_shape(conv6)
    x = Reshape(target_shape=LSTM_to_conv_dims, name='reshapelstmtoconv')(x)
    x = Lambda(lambda y: squeeze(y, 0))(x)
    print('将进酒', int_shape(x))

    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=(2, 2),
        padding='same',
        kernel_initializer=kernel_initializer)
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))(x)
    up1 = deconv_activation_layer(up1)
    batch7 = BatchNormalization(axis=-1)(up1)
    drop1 = Dropout(0.5)(batch7)
    print(int_shape(conv5))
    print(int_shape(drop1))
    merge1 = Concatenate(axis=-1)([conv5, drop1])
    #
    up2 = conv2d_transpose_factory(conv_n_filters[3], (5, 5))((merge1))
    up2 = deconv_activation_layer(up2)
    batch8 = BatchNormalization(axis=-1)(up2)
    drop2 = Dropout(0.5)(batch8)
    merge2 = Concatenate(axis=-1)([conv4, drop2])
    #
    up3 = conv2d_transpose_factory(conv_n_filters[2], (5, 5))((merge2))
    up3 = deconv_activation_layer(up3)
    batch9 = BatchNormalization(axis=-1)(up3)
    drop3 = Dropout(0.5)(batch9)
    merge3 = Concatenate(axis=-1)([conv3, drop3])
    #
    up4 = conv2d_transpose_factory(conv_n_filters[1], (5, 5))((merge3))
    up4 = deconv_activation_layer(up4)
    batch10 = BatchNormalization(axis=-1)(up4)
    merge4 = Concatenate(axis=-1)([conv2, batch10])
    #
    up5 = conv2d_transpose_factory(conv_n_filters[0], (5, 5))((merge4))
    up5 = deconv_activation_layer(up5)
    batch11 = BatchNormalization(axis=-1)(up5)
    merge5 = Concatenate(axis=-1)([conv1, batch11])
    #
    up6 = conv2d_transpose_factory(1, (5, 5), strides=(2, 2))((merge5))
    up6 = deconv_activation_layer(up6)
    batch12 = BatchNormalization(axis=-1)(up6)
    # Last layer to ensure initial shape reconstruction.
    up7 = Conv2D(1, (4, 4), dilation_rate=(2, 2), activation='sigmoid', padding='same', kernel_initializer=kernel_initializer)((batch12))
    output = multiply([up7, input_tensor])
    print(input_tensor.shape,output.shape)

    m = Model(inputs=input_tensor, outputs=output)
    return m

def trainModel(epochs=3, batch=1):
    m=apply_unet()
    m.compile(loss='mean_squared_error', optimizer='adam')

    model = m

    xTrain, yTrain = train()

    xValid, yValid = valid()

    model.fit(xTrain, yTrain, batch_size=batch, epochs=epochs, validation_data=(xValid, yValid))
    weightPath = '../weights' + ".h5"
    model.save_weights(weightPath, overwrite=True)

trainModel()
