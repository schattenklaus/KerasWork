#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:29:40 2017

@author: steffen
"""
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM, Dropout
# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length, 1)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 1000
# create LSTM
model = Sequential()
model.add(LSTM(32, input_shape=(length, 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(32, stateful=True, return_sequences=True))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
	print('%.1f' % value)


trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainY= np.reshape(trainY, (trainY.shape[0], 1, 1))
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))

testY=np.reshape(testY, (testY.shape[0], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))