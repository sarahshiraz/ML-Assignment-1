import numpy as np
from keras.datasets import mnist

def get_labels(num):
	'''
	This function takes a number from 0 - 9 and returns tha labels for classifying only that number
	e.g.
	get_labels(3) would make all the labels other than 3 zero. And all the labels for 3 equal to 1.
	'''
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	if num == 0:
		y_train[y_train!=0]=99
		y_train[y_train==0]=1
		y_train[y_train!=1]=0
	elif num ==1:
		y_train[y_train!=1]=0
	else:
		y_train[y_train!=num]=0
		y_train[y_train==num]=1
	return y_train

def cost_function_cross_entropy(pred, x, y):
	'''
	Takes predictions and ground truth and returns the binary cross entropy error between them 
	'''
	pred[pred == 0] = 0.0000000001
	pred[pred == 1] = 0.9999999999
	m = x.shape[0]
	a = y * np.log(pred+1e-9);
	b = (1 - y) * np.log( 1 - pred + 1e-9)
	if(np.isnan(np.sum(a))):
		a=0
	if(np.isnan(np.sum(b))):
		b=0
	total_cost = -(1 / m) * np.sum( a + b)
	return total_cost

def minibatch_stochastic(x_train, y_train, w, b, num_epochs, digit):
	'''
	Implements minibatch stochastic gradient descent for one classifer
	'''
	alpha = 0.001

	batch_size = 50;

	for num_epoch in range(num_epochs):
		shuffled_indices = np.random.permutation(m)
		x_shuffled = x_train[shuffled_indices]
		y_shuffled = y_train[shuffled_indices]
		z = np.dot(w,x_train.T) + b
		pred= 1/(np.exp(-z)+1)
		pred[pred >= 0.7] = 1 			#threshold for sigmoid set on 0.7.
		pred[pred<0.7] = 0

		dist = np.square((y_train - pred)).tolist()			# Find true positives and true negatives
		acc = dist.count(0)/len(dist);						# Dividing the sum of true positives and true negatives by total examples
		cost = cost_function_cross_entropy(pred,x_train,y_train) 	# find cost entropy error cost
		print("Classifer for digit: "+str(digit)+",		Epoch: " +str(num_epoch)+",     Accuracy: "+str(acc) +",     Cost: "+str(cost))
		for i in range(0, m, batch_size):
			xi = x_shuffled[i:i+batch_size]
			yi = y_shuffled[i:i+batch_size]
			z = np.dot(w,xi.T) + b
			pred = 1/(np.exp(-z)+1)
			pred[pred >= 0.7] = 1
			pred[pred<0.7] = 0
			temp = pred - yi
			for j in range(batch_size):
				grad = (alpha*temp[j]*xi[j]);			#calculating gradient
				w = w - grad;							# updating weights using each example
				b = b - (alpha*temp[j])

	return w,b,acc



(x_train, y_train), (x_test, y_test) = mnist.load_data()		#loading data
input_neurons = x_train[0].shape[0]*x_train[0].shape[1];
m = x_train.shape[0]
w = np.random.rand(10,input_neurons);
w = np.zeros((10,input_neurons))
b = np.random.rand(10)
b = np.zeros(10)
x_train_unrolled=[]
for x in x_train:
	x_train_unrolled.append((x.T).ravel());				# unrolling x_train
x_train_unrolled = np.array(x_train_unrolled)
x_train_unrolled = x_train_unrolled.astype('float32')
x_train_unrolled/=255							# normalizing x_train
x_test_unrolled=[]
for x in x_test:
	x_test_unrolled.append((x.T).ravel());				# unrolling x_test
x_test_unrolled = np.array(x_test_unrolled)
x_test_unrolled = x_test_unrolled.astype('float32')
x_test_unrolled/=255						# normalizing x_test
acc = np.zeros(10)
for i in range(10):
	y_train = get_labels(i)
	w[i,:], b[i], acc[i] = minibatch_stochastic(x_train_unrolled, y_train, w[i,:], b[i], 25, i)			# Applying minibatch stochastic gradient descent for all the classifiers


for j in range(10):
	print("Training Accuracy for classifier " + str(j) +": " + str(acc[j]))				# Prinitng accuracies for all the digit classifiers

'''Testing'''
x_test_unrolledT = x_test_unrolled.T
w = np.c_[np.ones((w.shape[0], 1)), w]
w[:,0] = b
x_test_unrolledT=np.r_[np.ones((1,x_test_unrolledT.shape[1])) , x_test_unrolledT]		


z = np.dot(w,x_test_unrolledT)				# Testing on learned weights
pred_all = 1/(np.exp(-z)+1)
pred_all[pred_all >= 0.7] = 1
pred_all[pred_all < 0.7] = 0
pred = np.argmax(pred_all , axis = 0)
dist = np.square((y_test - pred)).tolist()
acc = dist.count(0)/len(dist);
cost = cost_function_cross_entropy(pred,x_test_unrolled,y_test)
print( "Test Accuracy: "+str(acc) +",     Cost: "+str(cost))

