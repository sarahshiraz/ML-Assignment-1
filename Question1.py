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

def cost_function_sq_error(pred, y):
	'''
	Takes predictions and ground truth and returns the mean squared error between them 
	'''
	return np.sum(np.square((y - pred)))/(2*y.shape[0]);

def minibatch_stochastic(x_train, y_train, w, b, num_epochs, digit):
	'''
	Implements minibatch stochastic gradient descent for one classifer
	'''
	
	alpha = 0.001 		# learning rate

	batch_size = 500;

	for num_epoch in range(num_epochs):
		shuffled_indices = np.random.permutation(m)
		x_shuffled = x_train[shuffled_indices]
		y_shuffled = y_train[shuffled_indices]

		z = np.dot(w,x_train.T) + b

		pred= 1/(np.exp(-z)+1)
		pred[pred >= 0.7] = 1 		#threshold for sigmoid set on 0.7.
		pred[pred<0.7] = 0
		dist = np.square((y_train - pred)).tolist() 	# Find true positives and true negatives
		accuracy = dist.count(0)/len(dist);				# Dividing the sum of true positives and true negatives by total examples
		cost = cost_function_sq_error(pred,y_train)		# find mean squared error cost
		print("Classifer for digit: "+str(digit)+",		Epoch: " +str(num_epoch)+",    Training Accuracy: "+str(accuracy) +",    Training Cost: "+str(cost))
		for i in range(0, m, batch_size):
			xi = x_shuffled[i:i+batch_size]
			yi = y_shuffled[i:i+batch_size]
			z = np.dot(w,xi.T) + b
			pred = 1/(np.exp(-z)+1)
			pred[pred >= 0.7] = 1
			pred[pred<0.7] = 0
			temp = pred - yi
			pred = pred * (1 - pred)
			temp1 = temp * pred 				# Part of the gradient
			for j in range(batch_size):
				grad = temp[j]*pred[j]*xi[j]	# Multiplying with each feature to get corresponding weight
				w = w - alpha*grad 				# all weights updated using single example
				b = b - (temp[j]*pred[j])		# bias updated
	return w,b,accuracy



(x_train, y_train), (x_test, y_test) = mnist.load_data()
input_neurons = x_train[0].shape[0]*x_train[0].shape[1];
m = x_train.shape[0]
w = np.random.rand(10,input_neurons);
w = np.zeros((10,input_neurons))
b = np.random.rand(10)
b = np.zeros(10)
x_train_unrolled=[]
for x in x_train:					# unrolling x_train
	x_train_unrolled.append((x.T).ravel());
x_train_unrolled = np.array(x_train_unrolled)
x_train_unrolled = x_train_unrolled.astype('float32')
x_train_unrolled/=255				# normalizing x_train
x_test_unrolled=[]
for x in x_test:
	x_test_unrolled.append((x.T).ravel());			# unrolling x_test
x_test_unrolled = np.array(x_test_unrolled)
x_test_unrolled = x_test_unrolled.astype('float32')
x_test_unrolled/=255					# normalizing x_test
accuracy = np.zeros(10)
for i in range(10):
	y_train = get_labels(i)
	w[i,:], b[i], accuracy[i] = minibatch_stochastic(x_train_unrolled, y_train, w[i,:], b[i], 5, i)			# Applying minibatch stochastic gradient descent for all the classifiers


for j in range(10):
	print("Training Accuracy for classifier " + str(j) +": " + str(accuracy[j]))				# Prinitng accuracies for all the digit classifiers


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
accuracy = dist.count(0)/len(dist);
cost = cost_function_sq_error(pred,y_test)
print( "Test Accuracy: "+str(accuracy) +",     Cost: "+str(cost))


