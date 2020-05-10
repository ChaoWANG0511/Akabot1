from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Flatten,
    Reshape,
    TimeDistributed)
import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import Accuracy

from functools import partial
from keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    multiply,
    ReLU)
from keras.layers import Input
from keras.initializers import he_uniform



from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Flatten,
    Reshape,
    TimeDistributed)
#from tensorflow.train import AdamOptimizer
import tensorflow as  tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.metrics import Accuracy
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
import keras
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose  # merge, 
from tensorflow.keras.layers import MaxPool2D  #, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from tensorflow.keras.layers import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianDropout
from tensorflow import slice

import numpy as np
import numpy as np
import os
from tensorflow.keras.layers import Input, Reshape, Lambda
from tensorflow.keras.models import Model
from math import ceil

from functools import partial

import tensorflow as tf
from functools import partial
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    multiply,
    ReLU)

from tensorflow.keras.metrics import Accuracy
from tensorflow import slice


def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    smooth = 1.
    dropout_rate = 0.5
    act = "relu"

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', data_format='channels_first', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', data_format='channels_first', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x


def blstm():
    model = Sequential()
    # model.add(TimeDistributed(Flatten(input_shape=(x.shape[1:]))))
    # samples, time steps, features中的后两者
    model.add(Bidirectional(LSTM(250, input_shape=(30, 513), kernel_initializer=he_uniform(seed=50), return_sequences=True)))
    # return_sequences=true: 返回形如（samples，timesteps，output_dim）的3D张量
    # 通过LSTM，把词的维度由513转变成了250
    # BLSTM layer: 250 forward and 250 backward LSTM cells whose output is concatenated to form the overall output of the layer to form the overall output of the layer.
    model.add(Bidirectional(LSTM(250, kernel_initializer=he_uniform(seed=50), return_sequences=True)))
    model.add(Bidirectional(LSTM(250, kernel_initializer=he_uniform(seed=50), return_sequences=True)))
    model.add(TimeDistributed(Dense(513, activation='relu', kernel_initializer=he_uniform(seed=50))))
    # 每个time step都输出x.shape[2]长的向量
    # model.add(TimeDistributed(Reshape(x.shape[0:])))

    return model


def apply_unet():
    """ Apply a convolutionnal U-net to model a single instrument (one U-net
    is used for each instrument).
    :param input_tensor:
    :param output_name: (Optional) , default to 'output'
    :param params: (Optional) , default to empty dict.
    :param output_mask_logit: (Optional) , default to False.
    """
    conv_n_filters = [16, 32, 64, 128, 256, 512]
    conv_activation_layer = ReLU()
    deconv_activation_layer = ReLU()
    kernel_initializer = tf.keras.initializers.he_uniform(seed=50)
    conv2d_factory = partial(
        Conv2D,
        strides=(2, 2),
        padding='same',
        data_format='channels_first',
        kernel_initializer=kernel_initializer)

    #input_tensor = Input(shape=(None, None, 1), name='input')
    input_tensor = Input(shape=(2, 256, 2049), name='input')   #(512, 64, 1)
    b = slice(input_tensor, [0, 0, 0, 0], [-1, -1, -1, 2048])

    # First layer.
    conv1 = conv2d_factory(conv_n_filters[0], (5, 5))(b)
    batch1 = BatchNormalization(axis=-3)(conv1)
    rel1 = ReLU()(batch1)
    # Second layer.
    conv2 = conv2d_factory(conv_n_filters[1], (5, 5))(rel1)
    batch2 = BatchNormalization(axis=-3)(conv2)
    rel2 = ReLU()(batch2)
    # Third layer.
    conv3 = conv2d_factory(conv_n_filters[2], (5, 5))(rel2)
    batch3 = BatchNormalization(axis=-3)(conv3)
    rel3 = ReLU()(batch3)
    # Fourth layer.
    conv4 = conv2d_factory(conv_n_filters[3], (5, 5))(rel3)
    batch4 = BatchNormalization(axis=-3)(conv4)
    rel4 = ReLU()(batch4)
    # Fifth layer.
    conv5 = conv2d_factory(conv_n_filters[4], (5, 5))(rel4)
    batch5 = BatchNormalization(axis=-3)(conv5)
    rel5 = ReLU()(batch5)
    # Sixth layer
    conv6 = conv2d_factory(conv_n_filters[5], (5, 5))(rel5)
    batch6 = BatchNormalization(axis=-3)(conv6)
    _ = ReLU()(batch6)
    #
  
    conv2d_transpose_factory = partial(
        Conv2DTranspose,
        strides=(2, 2),
        padding='same',
        data_format='channels_first',
        kernel_initializer=kernel_initializer)
    #
    up1 = conv2d_transpose_factory(conv_n_filters[4], (5, 5))((conv6))
    up1 = ReLU()(up1)
    batch7 = BatchNormalization(axis=-3)(up1)
    drop1 = Dropout(0.5)(batch7)
    
    merge1 = Concatenate(axis=-3)([conv5, drop1])
    #
    up2 = conv2d_transpose_factory(conv_n_filters[3], (5, 5))((merge1))
    up2 = ReLU()(up2)
    batch8 = BatchNormalization(axis=-3)(up2)
    drop2 = Dropout(0.5)(batch8)
    merge2 = Concatenate(axis=-3)([conv4, drop2])
    #
    up3 = conv2d_transpose_factory(conv_n_filters[2], (5, 5))((merge2))
    up3 = ReLU()(up3)
    batch9 = BatchNormalization(axis=-3)(up3)
    drop3 = Dropout(0.5)(batch9)
    merge3 = Concatenate(axis=-3)([conv3, drop3])
    #
    up4 = conv2d_transpose_factory(conv_n_filters[1], (5, 5))((merge3))
    up4 = ReLU()(up4)
    batch10 = BatchNormalization(axis=-3)(up4)
    merge4 = Concatenate(axis=-3)([conv2, batch10])
    #
    up5 = conv2d_transpose_factory(conv_n_filters[0], (5, 5))((merge4))
    up5 = ReLU()(up5)
    batch11 = BatchNormalization(axis=-3)(up5)
    merge5 = Concatenate(axis=-3)([conv1, batch11])
    #
    up6 = conv2d_transpose_factory(1, (5, 5), strides=(2, 2))((merge5))
    up6 = ReLU()(up6)
    batch12 = BatchNormalization(axis=-3)(up6)
    # Last layer to ensure initial shape reconstruction.
    up7 = Conv2D(2, (4, 4), dilation_rate=(2, 2), activation='sigmoid', padding='same', data_format='channels_first', kernel_initializer=kernel_initializer)((batch12))
    output = multiply([up7, b])

    paddings = tf.constant([[0,0],[0,0],[0,0],[0,1]])
    output = tf.pad(output, paddings, "CONSTANT")

    m = Model(inputs=input_tensor, outputs=output)

    return m


def Nest_Net(img_rows=256, img_cols=2049, color_type=2, num_class=2, deep_supervision=False):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    #global bn_axis
    bn_axis = 1
    img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')
    input_2048 = slice(img_input, [0, 0, 0, 0], [-1, -1, -1, 2048])

    conv1_1 = standard_unit(input_2048, stage='11', nb_filter=nb_filter[0])  # 大小不变：96，32 filters
    pool1 = MaxPool2D((2, 2), strides=(2, 2), name='pool1', data_format='channels_first')(conv1_1)  # 大小：96变48

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])   # 大小不变：48，64 filters
    pool2 = MaxPool2D((2, 2), strides=(2, 2), name='pool2', data_format='channels_first')(conv2_1)   # 大小：48变24

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same', data_format='channels_first')(conv2_1)  #
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)  #
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])   #

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])  #
    pool3 = MaxPool2D((2, 2), strides=(2, 2), name='pool3', data_format='channels_first')(conv3_1)  #

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same', data_format='channels_first')(conv3_1)  #
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)    #
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])  #

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same', data_format='channels_first')(conv2_2)   #
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)  #
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])   #

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])  #
    pool4 = MaxPool2D((2, 2), strides=(2, 2), name='pool4', data_format='channels_first')(conv4_1)   #

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same', data_format='channels_first')(conv4_1)  #
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)   #
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])   #

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same', data_format='channels_first')(conv3_2)   #
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)  #
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])  #

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same', data_format='channels_first')(conv2_3)   #
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)  #
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])  #

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])   #

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same', data_format='channels_first')(conv5_1)   #
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)  #
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])   #

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same', data_format='channels_first')(conv4_2)   #
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)  #
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])   #

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same', data_format='channels_first')(conv3_3)   #
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)    #
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])    #

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same', data_format='channels_first')(conv2_4)   #
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)   #
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])  #

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4), data_format='channels_first')(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4), data_format='channels_first')(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4), data_format='channels_first')(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4), data_format='channels_first')(conv1_5)

    paddings = tf.constant([[0,0],[0,0],[0,0],[0,1]])
    output = tf.pad(nestnet_output_4, paddings, "CONSTANT")

    model = Model(inputs=img_input, outputs=output)

    return model


def apply_blstm(d_model=(2, 256, 2049), name='blstm', lstm_units=250):
    """ Apply BLSTM to the given input_tensor.
    :param input_tensor: Input of the model.
    :param output_name: (Optional) name of the output, default to 'output'.
    :param params: (Optional) dict of BLSTM parameters.
    :returns: Output tensor.
    """
    inputm = tf.keras.Input(shape=d_model, name="input")
    inputt = tf.transpose(inputm, perm=[0, 2, 1, 3])
    #slices = np.transpose(slices, (0, 2, 1, 3)) # 想要[batch, time, channel, freaquency] 

    units = lstm_units
    kernel_initializer = he_uniform(seed=50)
    flatten_input = TimeDistributed(Flatten())((inputt))

    def create_bidirectional():
        return Bidirectional(
            CuDNNLSTM(
                units,
                kernel_initializer=kernel_initializer,
                return_sequences=True))

    l1 = create_bidirectional()((flatten_input))
    l2 = create_bidirectional()((l1))
    l3 = create_bidirectional()((l2))
    dense = TimeDistributed(
        Dense(
            int(flatten_input.shape[2]),
            activation='relu',
            kernel_initializer=kernel_initializer))((l3))
    outputt = TimeDistributed(
        Reshape(inputt.shape[2:]),
        name='output')(dense)
    outputm = tf.transpose(outputt, perm=[0, 2, 1, 3])
    return tf.keras.Model(inputs=inputm, outputs=outputm, name=name)
