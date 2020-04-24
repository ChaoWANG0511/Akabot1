#!/usr/bin/env python
# -*- coding:utf-8 -*-

from tensorflow.keras.layers import GRU, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
def gru():
    model = Sequential()
    model.add(Bidirectional(GRU(32, dropout=0.1,recurrent_dropout=0.5,return_sequences=True, input_shape=(30, 513))))
    model.add(Bidirectional(GRU(64, activation='relu',dropout=0.1,recurrent_dropout=0.5,return_sequences=True)))
    model.add(Activation('sigmoid'))
    model.add(TimeDistributed(Dense(513, activation='relu', kernel_initializer=he_uniform(seed=50))))
    return model
