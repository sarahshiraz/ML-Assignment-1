import numpy as np
from keras.datasets import mnist
import collections
import keras


def cost_function_binary_cross_entropy(a, y):
	m = y.shape[1];
	cost = -1* np.sum(np.sum(y * np.log(a + 1e-9), axis=0))/m;
	return cost;

def minibatch(x_train, y_train, w, b, num_epochs):
	
	alpha = 0.001

	batch_size = 50;

	m = x_train.shape[0]

	for num_epoch in range(num_epochs):
		shuffled_indices = np.random.permutation(m)
		x_shuffled = x_train[shuffled_indices]
		y_shuffled = y_train[shuffled_indices]

		z = np.dot(w,x_train.T) + b.T
		expz = np.exp(z) # Take exponent of all elements
		exp_z_sum = np.sum(expz, axis=0); # Take exponent of all elements and sum them column wise to get a 1x60000 vector
		a_softmax = expz/exp_z_sum; # Divide each element with the sum of all column i.e. exponent of all neuron z values.

		pred= np.argmax(a_softmax, axis = 0)

		y_normal = np.argmax(y_train.T, axis = 0)

		total_correct = collections.Counter(pred==y_normal)[1]

		m = x_train.shape[0]
		acc = total_correct/60000;
		cost = cost_function_binary_cross_entropy(a_softmax,y_train.T)
		print("Epoch: " +str(num_epoch)+",     Accuracy: "+str(acc) +",     Cost: "+str(cost))
		for i in range(0, 60000, batch_size):
			xi = x_shuffled[i:i+batch_size]
			yi = y_shuffled[i:i+batch_size]

			'''check if yi is right'''
			z = np.dot(w,xi.T) + b.T
			expz = np.exp(z) # Take exponent of all elements
			exp_z_sum = np.sum(expz, axis=0) # Take exponent of all elements and sum them column wise to get a 1x60000 vector
			a_softmax = expz/exp_z_sum; # Divide each element with the sum of all column i.e. exponent of all neuron z values.

			yiT=yi.T

			for j in range(batch_size):
				partialLbyPartialZ = np.zeros(10)
				for n in range(10):
					am = a_softmax[n,j];
					yl = yiT[:,j]
					for l in range(10):
						delta = 0
						if l == n:
							delta = 1
						partialLbyPartialZ[n] = partialLbyPartialZ[n] + (yl[l] * (am - delta))
				for p in range(10):
					grad = (alpha*partialLbyPartialZ[p]*xi[j]);
					w[p,:] = w[p,:] - grad;
					b[0,p] = b[0,p] - alpha*partialLbyPartialZ[p];
	return w,b,acc



(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_neurons = x_train[0].shape[0]*x_train[0].shape[1];
m = x_train.shape[0]
w = np.random.rand(10,input_neurons);
w = np.zeros((10,input_neurons))
b = np.random.rand(10)
b = np.zeros((1,10))



x_train_unrolled=[]
for x in x_train:
	x_train_unrolled.append((x.T).ravel());
x_train_unrolled = np.array(x_train_unrolled)
x_train_unrolled = x_train_unrolled.astype('float32')
x_train_unrolled/=255
x_test_unrolled=[]
for x in x_test:
	x_test_unrolled.append((x.T).ravel());
x_test_unrolled = np.array(x_test_unrolled)
x_test_unrolled = x_test_unrolled.astype('float32')
x_test_unrolled/=255
acc = 0
num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

w, b, acc = minibatch(x_train_unrolled, y_train, w, b, 20)

'''Testing'''
z = np.dot(w,x_test_unrolled.T) + b.T
expz = np.exp(z) # Take exponent of all elements
exp_z_sum = np.sum(expz, axis=0); # Take exponent of all elements and sum them column wise to get a 1x60000 vector
a_softmax = expz/exp_z_sum; # Divide each element with the sum of all column i.e. exponent of all neuron z values.

pred= np.argmax(a_softmax, axis = 0)
y_normal = np.argmax(y_test.T, axis = 0)
total_correct = collections.Counter(pred==y_normal)[1]
print("total_correct")
print(total_correct)
m = x_test_unrolled.shape[0]
print("m")
print(m)
acc = total_correct/m;
cost = cost_function_binary_cross_entropy(a_softmax,y_test.T)
print("Test Accuracy: "+str(acc) +",    Test Cost: "+str(cost))
