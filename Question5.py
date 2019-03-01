'''

RUN countWhiteRegions.py file before running this
countWhiteRegions.py counts number of white regions in all the training and testing examples and saves in two npy files	

'''


import numpy as np
import collections
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import optimizers

batch_size = 128
num_classes = 10
epochs = 100

(x_train, y_train), (x_test, y_test) = mnist.load_data()

conn_comp_train = np.load('conn_comp_train.npy')		# Getting the vector of the number of white regions for all training images
conn_comp_test = np.load('conn_comp_test.npy')			# Getting the vector of the number of white regions for all testing images

conn_comp_train = conn_comp_train.astype('float32')
conn_comp_train/=3										# Normalizing for be between zero and 1

conn_comp_test = conn_comp_test.astype('float32')
conn_comp_test/=3										# Normalizing for be between zero and 1

conn_comp_train = conn_comp_train.reshape(1,conn_comp_train.shape[0])     
conn_comp_test = conn_comp_test.reshape(1,conn_comp_test.shape[0])		  


x_train_unrolled=[]
for x in x_train:
	x_train_unrolled.append((x.T).ravel());				# Unrolling x_train
x_train_unrolled = np.array(x_train_unrolled)		
x_train_unrolled = x_train_unrolled.astype('float32')
x_train_unrolled/=255				# normalizing x_train

x_test_unrolled=[]
for x in x_test:
	x_test_unrolled.append((x.T).ravel());			# Unrolling x_test
x_test_unrolled = np.array(x_test_unrolled)
x_test_unrolled = x_test_unrolled.astype('float32')
x_test_unrolled/=255			# normalizing x_test


x_train_unrolled = (np.vstack((x_train_unrolled.T, conn_comp_train))).T 			# Concatenating the vector for training images with training data
x_test_unrolled = (np.vstack((x_test_unrolled.T, conn_comp_test))).T 				# Concatenating the vector for testing images with testing data

y_train = keras.utils.to_categorical(y_train, num_classes) 		# converting labels to hot encoding	
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(num_classes, activation='softmax', input_shape = (785,))) 	# one fully connected layer

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9) 		# optimizer parameters
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train_unrolled, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test_unrolled, y_test))  		# training

score = model.evaluate(x_test_unrolled, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
