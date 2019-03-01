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

(x_train, y_train), (x_test, y_test) = mnist.load_data()		# load data

x_train_unrolled=[]
for x in x_train:
	x_train_unrolled.append((x.T).ravel());
x_train_unrolled = np.array(x_train_unrolled)				# unrolling each x_train sample from 28x28 to 784x1
x_train_unrolled = x_train_unrolled.astype('float32')
x_train_unrolled/=255			# normalizing x_train

x_test_unrolled=[]
for x in x_test:
	x_test_unrolled.append((x.T).ravel());				# unrolling each x_test sample from 28x28 to 784x1	
x_test_unrolled = np.array(x_test_unrolled)
x_test_unrolled = x_test_unrolled.astype('float32')
x_test_unrolled/=255				# normalizing x_test


y_train = keras.utils.to_categorical(y_train, num_classes) 	# converting labels to hot encoding	
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(num_classes, activation='softmax', input_shape = (784,))) 		# One fully connected layer

sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)			# Defining optimizer parameters
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train_unrolled, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test_unrolled, y_test)) 		# Training

score = model.evaluate(x_test_unrolled, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
