#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:52:15 2017

@author: steffen
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Input, Dense, Lambda , TimeDistributed, LSTM , Dropout
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from edward.util import Progbar


run=2
np.random.seed(7)
plt.style.use(['seaborn-darkgrid'])
batch_size = 2   #How many values of my data do i look on
look_back = 9    #BPTT, how many steps I look back in time
featurenumber=1  #If i have different features
latent_dim = 32


epochs=100
np.random.seed(7)

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


if run==1:
   #TOY DATASET
   dataset = np.cos(np.arange(1000)*(20*np.pi/1000))[:,None]
   plt.plot(dataset)
   #scaler = MinMaxScaler(feature_range=(0, 1))
#   dataset = scaler.fit_transform(dataset)
   print "Dataset", run
elif run==2:
   dataset = read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
   dataset = dataset.values
   dataset = dataset.astype('float32')
   print "Dataset", run
   plt.plot(dataset)
   scaler = MinMaxScaler(feature_range=(0, 1))
   dataset = scaler.fit_transform(dataset)




train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# Encoder

inputs = Input(shape=(batch_size, look_back, featurenumber))
encoded = LSTM(latent_dim, stateful=True)(inputs)

outputs = LSTM(latent_dim, stateful=True)(inputs)

AE = Model(inputs, outputs)







