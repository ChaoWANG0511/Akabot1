from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense

from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import Reshape


inputs = Input(shape=(784,))                 # input layer
x = Dense(32, activation='relu')(inputs)     # hidden layer
outputs = Dense(10, activation='softmax')(x) # output layer

model = Model(inputs, outputs)



def apply_blstm():

    mashup = Input(shape=(128, 128), name='input')  # shape不含batch size, None意味着可以随便取
    convA = Conv2D(64, 3, activation='relu', padding='same')(mashup)

    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_shape=(128, 128), return_sequences=True)))
    model.add(Bidirectional(LSTM(12, return_sequences=True)))
    model.add(Bidirectional(LSTM(6, return_sequences=False)))
    model.add(Dense(128 * 128))
    model.add(Reshape((128, 128)))