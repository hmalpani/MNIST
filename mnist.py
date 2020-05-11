import numpy as np

import h5py
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import np_utils
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.python.framework import ops
import math
import scipy
from PIL import Image
from scipy import ndimage


#print(mnist.load_data())
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.astype(np.float32)
#x_train = np.expand_dims(x_train,-1)

#x_train = x_train.reshape(x_train.shape[0],-1).T
#y_train = tf.one_hot(y_train,10,axis=0)
#y_train = np.array(y_train,'float32')

x_train = x_train.reshape(x_train.shape[0], -1).T
x_test = x_test.reshape(x_test.shape[0], -1).T

x_train = x_train/255.
x_test = x_test/255.

y_train = tf.one_hot(y_train, 10,axis=0)
y_test = tf.one_hot(y_test, 10,axis=0)

sess=tf.compat.v1.Session()
y_train=sess.run(y_train)
y_test=sess.run(y_test)

print ("number of training examples = " + str(x_train.shape[1]))
print ("number of test examples = " + str(x_test.shape[1]))
print ("X_train shape: " + str(x_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(x_test.shape))
print ("Y_test shape: " + str(y_test.shape))


def create_placeholder(n_x,n_y):
	X=tf.placeholder(tf.float32, shape=(n_x,None), name="X")
	Y=tf.placeholder(tf.float32, shape=(n_y,None), name="Y")
	return X,Y
	
def initialize_parameters():
	W1 = tf.get_variable("W1", [25,784],initializer=tf.contrib.layers.xavier_initializer())
	b1 = tf.get_variable("b1", [25,1],initializer=tf.zeros_initializer())
	W2 = tf.get_variable("W2", [15,25],initializer=tf.contrib.layers.xavier_initializer())
	b2 = tf.get_variable("b2", [15,1],initializer=tf.zeros_initializer())
	W3 = tf.get_variable("W3", [10,15],initializer=tf.contrib.layers.xavier_initializer())
	b3 = tf.get_variable("b3", [10,1],initializer=tf.zeros_initializer())
	
	parameters = {"W1":W1,
				  "b1":b1,
				  "W2":W2,
				  "b2":b2,
				  "W3":W3,
				  "b3":b3}
				  
	return parameters
	

def forward_propagation(X,parameters):

	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
    
	Z1 = tf.add(tf.matmul(W1,X),b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2,A1),b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3,A2),b3)
    
	return Z3
    
def compute_cost(Z3,Y):
	logits = tf.transpose(Z3)
	labels = tf.transpose(Y)
    
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
    
	return cost
    
def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]#.reshape((1, m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (k + 1) * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, (k + 1) * mini_batch_size:]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
    
def model(x_train, y_train, x_test, y_test, learning_rate=0.0001, num_epochs=500, minibatch_size=64, print_cost=True):
	ops.reset_default_graph()
	(n_x, m) = x_train.shape
	n_y = y_train.shape[0]
	costs = []
	
	X, Y = create_placeholder(n_x,n_y)
	parameters = initialize_parameters()
	Z3 = forward_propagation(X,parameters)
	cost = compute_cost(Z3,Y)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	init = tf.global_variables_initializer()
	
	
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			epoch_cost = 0.
			num_minibatches=int(m/minibatch_size)
			minibatches = random_mini_batches(x_train, y_train, minibatch_size)
			print(epoch)
			for minibatch in minibatches:
				(minibatch_X,minibatch_Y)=minibatch
				_ , minibatch_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
				epoch_cost += minibatch_cost / minibatch_size
				
			
			if print_cost == True and epoch % 100 == 0:
				print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
				correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
				print("Accuracy: ", accuracy.eval({X: x_train, Y: y_train}))
				print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
			if print_cost == True and epoch % 5 == 0:
				costs.append(epoch_cost)

                
		parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#		print ("Train Accuracy:", accuracy.eval({X: x_train, Y: y_train}))
#		print ("Test Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
        
        return parameters
        
        
        
parameters = model(x_train,y_train,x_test,y_test)
        
print(parameters)
        
def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))
            
            
save_dict_to_hdf5(parameters,"parameters.h5")
        

