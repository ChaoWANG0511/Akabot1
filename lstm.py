import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D, Concatenate
from keras.models import Model


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

def blstm():

    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_shape = (128,128), return_sequences=True)))
    model.add(Bidirectional(LSTM(12, return_sequences=True)))
    model.add(Bidirectional(LSTM(6, return_sequences=False)))
    model.add(Dense(128*128))
    model.add(Reshape(128,128))

    return m














