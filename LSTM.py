#!/usr/bin/python

import sys
from time import time
sys.path.append("E:/PythonProjects/DeepLearning/")
from Wikicrawler import preprocessing

import io
import os

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

title = preprocessing()
print(title)
#print os.path.abspath("cleanText.txt")
f = open("E:/PythonProjects/DeepLearning/cleanText.txt",encoding='utf-8',errors='ignore')
text = f.read()

chars = sorted(list(set(text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))


# print( len(chars), len(text))

seq_length = 100
dataX = []
dataY = []
for i in range(0,len(text) - seq_length,1):
    seq_in = text[i:i+seq_length]
    seq_out = text[i+seq_length]
    #mapping integer values using dictionary char_to_int
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[seq_out]])

# print(len(dataX))

x = np.reshape(dataX,(len(dataX),seq_length,1))
# print(x.shape)

#normalizing (o to 1)
x = x/len(chars)
#one hot encoding 
y = np_utils.to_categorical(dataY)

##############
##LSTM Model
##############

model = Sequential()
model.add(LSTM(256,input_shape=(x.shape[1],x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

#checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbackList = [checkpoint]

#model fitting
model.fit(x,y,batch_size=128,epochs=20,callbacks=callbackList)










