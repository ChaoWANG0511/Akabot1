# build CNN/LSTM and train it.
#
model = Sequential()

# build CNN/LSTM and train it.

model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same'), batch_input_shape=(32, None, None, 1)))
model.add(Activation('elu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(Dropout(0.2))

model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same')))
model.add(Activation('elu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model.add(Dropout(0.2))

model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same')))
model.add(Activation('elu'))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

model.add(TimeDistributed(Flatten()))

model.add(Conv1D(16, 3, padding='same'))
model.add(Activation('elu'))
model.add(MaxPooling1D(pool_size=8))

model.add(Bidirectional(LSTM(64, return_sequences=True, stateful=True)))
model.add(Activation('elu'))
model.add(Bidirectional(LSTM(128, return_sequences=False, stateful=True)))
model.add(Activation('elu'))
model.add(Dense(1, activation='sigmoid'))
adammgm = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
model.compile(loss='mean_squared_error', optimizer=adammgm, metrics=['accuracy'])
print(model.summary())