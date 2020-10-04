# Load LSTM network and generate text
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

f = open("E:/PythonProjects/DeepLearning/cleanText.txt",encoding='utf-8',errors='ignore')
text = f.read()

chars = sorted(list(set(text)))
char_to_int = dict((c,i) for i,c in enumerate(chars))
int_to_char = dict((i,c) for i,c in enumerate(chars))

seq_length = 100
dataX = []
dataY = []

for i in range(0,len(text) - seq_length,1):
    seq_in = text[i:i+seq_length]
    seq_out = text[i+seq_length]
    #mapping integer values using dictionary char_to_int
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append([char_to_int[seq_out]])

x = np.reshape(dataX,(len(dataX),seq_length,1))
# print(x.shape)

#normalizing (o to 1)
x = x/float(len(chars))
#one hot encoding 
y = np_utils.to_categorical(dataY)

##############
##LSTM Model loading
##############

model = Sequential()
model.add(LSTM(256,input_shape=(x.shape[1],x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))

filename = "E:/PythonProjects/DeepLearning/model/weights-improvement-17-0.5166.hdf5"

model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

#seeding
start = np.random.randint(0,len(dataX)-1)
pattern = dataX[start]
print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")

generatedText = ""

for i in range(1000):
    x = np.reshape(pattern,(1,len(pattern),1))
    x = x/float(len(chars))
    prediction = model.predict(x,verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    generatedText += result
    seq_in = [int_to_char[val] for val in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    
def sendText():
    return generatedText