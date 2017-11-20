#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:15:49 2017

@author: steffen
"""
import os, random
import numpy             as np
import matplotlib.pyplot as plt
import keras.backend     as K
import tensorflow as tf


from keras.datasets import mnist
from keras.models   import Sequential
from keras.layers   import Dense, Reshape
from keras.utils                import np_utils
from keras.optimizers           import Adam
from keras.layers.core          import Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU

##############################################################################
np.random.seed(777)
##############################################################################
# Load Sample Data
img_rows, img_cols = 28, 28
K.image_data_format()
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(-1, 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    print "Ordering Tensorflow Style"
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    print "Ordering Theano Style"



X_train = X_train.astype('float32')
X_test  = X_test.astype ('float32')
X_train = X_train / 255
X_test  = X_test  /255


##############################################################################


##############################################################################
def build_genenerator():
    G=Sequential()

    G.add(Dense(200*14*14,  input_dim=100, init='glorot_normal'))
    G.add(BatchNormalization(mode=0))
    G.add(Activation("relu"))
    G.add(Reshape((200, 14, 14), input_shape=(200*14*14,)))

    G.add(UpSampling2D(size=(2, 2)))
    G.add(Convolution2D(100, 3, 3, border_mode='same', init='glorot_uniform'))
    G.add(BatchNormalization(mode=0))
    G.add(Activation('relu'))

    G.add(Convolution2D(25, 3, 3, border_mode='same', init='glorot_uniform'))
    G.add(BatchNormalization(mode=0))
    G.add(Activation('relu'))

    G.add(Convolution2D(1, 3, 3, border_mode='same', init='glorot_uniform'))
    G.add(Activation('sigmoid'))
    return G


D=Sequential()

D.add(Convolution2D(32 , 3, 3, subsample=(2,2), border_mode="same",
                        input_shape=(1,28,28)))
D.add(LeakyReLU(alpha=0.2))

D.add(Dropout(0.2))
D.add(Convolution2D(64, 3, 3, subsample=(2,2)))
D.add(LeakyReLU(alpha=0.2))

D.add(Dropout(0.2))
D.add(Flatten())

D.add(Dense(512))
D.add(LeakyReLU(alpha=0.2))

D.add(Dense(11))
D.add(Activation('softmax'))


##############################################################################
# Load Instances of descriptor and generator
discriminator = D
generator  = build_genenerator()

dopt=Adam(lr=1e-3)
gopt=Adam(lr=1e-4)

# Build Discriminator Model
D_Modell = Sequential()
D_Modell.add(discriminator)
D_Modell.compile(loss='categorical_crossentropy',
                 optimizer=dopt)

# Build Adversarial Modell: D + G --> A
discriminator.trainable = False
A_Modell = Sequential()
A_Modell.add(generator)
A_Modell.add(discriminator)
A_Modell.compile(loss='categorical_crossentropy',
                 optimizer=gopt)

##############################################################################
def get_ytrue(batch_size):
    y = np.zeros(2*batch_size)
    y[0:batch_size] = 1
    return y

def get_random_noise(batch_size):
    return np.random.uniform(0, 1,size=[batch_size,100])

def get_traningset(X,Y,batch_size):
    _id=np.random.randint(0,X.shape[0],size=batch_size)
    image_batch = X[_id]
    generated_images = generator.predict(get_random_noise(batch_size))
    x = np.concatenate((image_batch , generated_images))
    y = np.zeros(2*batch_size)
    y[0:batch_size] = Y[_id]
    y[batch_size:2*batch_size] = 10
    y = np_utils.to_categorical(y, 11)
    return x,y

def pretrain_D(batch_size, X, Y):
    for k in range(200):
        x,y = get_traningset(X,Y,batch_size)
        D_Modell.train_on_batch(x, y)
    print "D pretrained"


##############################################################################
Pre_batch_size = 64
loss           = {"D":[], "A": []}

# Pretraining discriminator resulted in poor performance
#pretrain_D(Pre_batch_size, X_train,y_train)
#pretrain_D(Pre_batch_size, X_train)

##############################################################################

def train_alternating_CAT(X,Y, nb_epoch=5, batch_size=128):
    for epoch in range(nb_epoch):
        print "Epoch: "+str(epoch)

        # Load Data and train Descriptor
        x,y = get_traningset(X,Y,batch_size)
        d_loss = D_Modell.train_on_batch(x, y)
        loss["D"].append(d_loss)

        # Create Noise and train Adversarial Network
        noise   = get_random_noise(batch_size)
        y_true  = np.ones(batch_size)*7 # Set here Target of generator
        y_true  = np_utils.to_categorical(y_true, 11)

        a_loss = A_Modell.train_on_batch(noise,y_true)
        loss["A"].append(a_loss)

        if epoch%5==0:
            generated_images = generator.predict(get_random_noise(1))
            plt.figure()
            plt.imshow(generated_images[0][0])
            plt.savefig("gan_progress//large_config_7_"+str(t)+"_epoch_"+str(epoch)+".png")
            plt.close()

        # Printing Single-layer from g --> checking if gradient!=0
        print "G updated"
        print A_Modell.layers[0].layers[0].get_weights()[0][0][0:5]

##############################################################################
t=0
dopt=Adam(lr=1e-4)
gopt=Adam(lr=1e-5)
train_alternating_CAT(X_train,y_train, nb_epoch=1000, batch_size=128)

dopt=Adam(lr=1e-5)
gopt=Adam(lr=1e-6)
t+=1
train_alternating_CAT(X_train,y_train, nb_epoch=500, batch_size=128)
K.set_value(dopt.lr,1e-6)
K.set_value(gopt.lr,1e-7)
t+=1
train_alternating_CAT(X_train,y_train, nb_epoch=500, batch_size=128)

plt.figure()
plt.plot(loss["D"],label="D")
plt.plot(loss["A"],label="A")
plt.legend()
plt.grid()