import numpy as np
import math
import cv2
from numpy import asarray
from imutils import paths
import argparse
import os
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
from PIL import Image
from scipy import ndimage





def sigmoid(Z):
	A = 1. / (1 + np.exp(-Z))
	return A




def relu(Z):
	A = np.maximum(np.zeros(Z.shape),Z)
	return A




def tanh(Z):
	A = np.tanh(Z)
	return A



def softmax(Z):
    C = -np.max(Z) #set constant to prevent denom from blowing up.
    t = np.exp(Z + C)
    A = t / np.sum(t, axis=0)
    return A




def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1-AL).T)) / m
    return np.squeeze(cost)




def compute_softmax_cost(AL, Y):
	m = Y.shape[1]
	L = np.sum(np.log(AL) * Y, axis=0)
	return -(1./m)*np.sum(L)
    
    
    
    
def load_datasets(root, height, width, train_ratio=0.7, gray=True):
	
	if root[len(root)-1] != "/" : root = root + "/"
	dirs = os.listdir(root)
	
	dataset_size = len(list(paths.list_images(root)))
	
	trainset_size = 0
	for subset in dirs:
		imagePaths = list(paths.list_images(root+subset))
		m = len(imagePaths)
		trainset_size += int(m*float(train_ratio))
	
	testset_size = dataset_size - trainset_size
	print("dataset size = " + str(dataset_size))
	print("training set size = " + str(trainset_size))
	print("test set size = " + str(testset_size))
	
	X_train = np.ndarray(shape=(trainset_size, height, width), dtype=np.uint8) 
	Y_train = np.ndarray(shape=(trainset_size, 1), dtype=int)
	X_test = np.ndarray(shape=(testset_size, height, width), dtype=np.uint8)
	Y_test = np.ndarray(shape=(testset_size, 1), dtype=int)
	
	label = 1
	i = 0
	k = 0 
	
	for subset in dirs:	
		print("Subset '{subset}' has label {label}".format(subset=subset, label=label))
		imagePaths = list(paths.list_images(root+subset))
		m = len(imagePaths)
		n_train = int(m*float(train_ratio))

		for j in range(m): 

			img = np.array(ndimage.imread(imagePaths[j], flatten=True))	
			
			if img.shape != (height, width, 3): 
				img = cv2.resize(img, (height, width))
			if j < n_train:
				X_train[i] = img
				Y_train[i] = label
				i+=1		
			else: # Got enough training examples. Get test examples.
				X_test[k] = img
				Y_test[k] = label
				k += 1
		label += 1

	return X_train, Y_train, X_test, Y_test 






def flatten_dataset(X_train_orig, X_test_orig):
	m_train = X_train_orig.shape[0]
	h_train = X_train_orig.shape[1]
	w_train = X_train_orig.shape[2]
	
	m_test = X_test_orig.shape[0]
	h_test = X_test_orig.shape[1]
	w_test = X_test_orig.shape[2]

	X_train = X_train_orig.reshape(h_train*w_train, m_train)
	X_test = X_test_orig.reshape(h_test*w_test, m_test)
	
	return X_train, X_test



	
	
def one_hot_encoder(Y_orig, classes):
	Y = np.zeros((classes, len(Y_orig)))
	for i in range(len(Y_orig)):
		Y[Y_orig[i]-1,i] = 1
	return Y
	
	
	
	
def initialize_params(layer_dims):  # initialize weights (randomly) and biases (zeros)
	params = {}
	for l in range(len(layer_dims)-1):
		params["W" + str(l+1)] = np.random.randn(layer_dims[l+1], layer_dims[l]) * 0.01 # scale to prevent explosion
		params["b" + str(l+1)] = np.zeros((layer_dims[l+1], 1))
	return params

def initialize_params_Xavier(layer_dims):
	params = {}
	for l in range(len(layer_dims)-1):
		params["W" + str(l+1)] = np.random.randn(layer_dims[l+1], layer_dims[l]) * np.sqrt(1./layer_dims[l])
		params["b" + str(l+1)] = np.zeros((layer_dims[l+1], 1))
	return params
	
def initialize_params_He(layer_dims):
	params = {}
	for l in range(len(layer_dims)-1):
		params["W" + str(l+1)] = np.random.randn(layer_dims[l+1], layer_dims[l]) * np.sqrt(2./layer_dims[l])
		params["b" + str(l+1)] = np.zeros((layer_dims[l+1], 1)) 
	return params
	





def random_mini_batches(X_train, Y_train, batch_size):
	
	#shuffle examples
	assert len(X_train) == len(Y_train)
	shuffler = np.random.permutation(len(X_train))
	X_train = X_train[shuffler]
	Y_train = Y_train[shuffler]
	
	# create minibatches
	n = math.ceil(len(X_train) / batch_size)
	#TODO: divide X_train into n minibatches 
	return




def forward_prop(params, X_train, Y_train, layer_dims, activations):

	# intialize cache dictionary for saving activations
	L = len(layer_dims)
	cache = {}
	cache["A0"] = X_train 
    	
	# iterate over layer_dims
	for l in range(L - 1):
		cache["Z" + str(l+1)] = np.dot(params["W" + str(l+1)], cache["A" + str(l)]) + params["b" + str(l+1)]
    		if activations[l] == "sigmoid":
    			cache["A" + str(l+1)] = sigmoid(cache["Z"+ str(l+1)])
    		elif activations[l] == "ReLU":
    			cache["A" + str(l+1)] = relu(cache["Z"+ str(l+1)])
    		elif activations[l] == "tanh":
    			cache["A" + str(l+1)] = tanh(cache["Z"+ str(l+1)])
		elif activations[l] == "softmax":
    			cache["A" + str(l+1)] = softmax(cache["Z"+ str(l+1)])
    	
    	if activations[L-2] == "softmax":			
		cost = compute_softmax_cost(cache["A" + str(L-1)], Y_train)
	else:
		cost = compute_cost(cache["A" + str(L-1)], Y_train)
	
    	return cache, cost

def forward_prop_with_dropout():
    None
    return cache, cost

def forward_prop_with_regularization():
    None
    return cache, cost
 


def compute_cost_with_regularization():
    None
    return


def gradient_check(grads, params, X, Y, layer_dims, activations, epsilon = 1e-7): #after backprop & before updating params
	# combine weight and bias vectors into vector theta
	theta_plus = {}
	theta_minus = {}
	
	dtheta_approx = np.array([[]])
	dtheta = np.array([[]])
	
	L = len(params) // 2
	
	#create vector theta from param values
	#for i in range(1, L+1):
	#	Wi = params["W" + str(i)].reshape(1,-1)
	#	bi = params["b" + str(i)].reshape(1,-1)
	#	theta_i = np.concatenate((Wi, bi), axis=1)
	#	theta = np.concatenate((theta, theta_i), axis=1)	
	
	#compute approximate gradient
	#for i in range(1, len(theta)+1):
	#	thetaplus = theta
	#	thetaplus[i] += epsilon
			
		#convert vector theta back to params dictionary
	#	for i in range (1, L+1):
	#		Wi = 
	#		params["W" + str(i)] = 
	#		bi = params["b" + str(i)].reshape(1,-1)
		
	#compute approx. dtheta
	i = 0
	for p in list(params.keys()):	
		for index, _ in np.ndenumerate(params[p]):

			theta_plus = params
			theta_plus[p][index] += epsilon
			_, J_plus = forward_prop(theta_plus, X, Y, layer_dims, activations)
			#print("J_plus: " + str(J_plus))
			
			theta_minus = params
			theta_minus[p][index] -= epsilon
			_, J_minus = forward_prop(theta_minus, X, Y, layer_dims, activations)
			#print("J_minus: " + str(J_minus))
			
			dtheta_approx = np.append( dtheta_approx, (J_plus - J_minus) / (2*epsilon))
			#print("dtheta_approx: " + str(dtheta_approx[i]))
			dtheta = np.append( dtheta, grads["d" + p][index])
			#print("dtheta: " + str(dtheta[i]))
			#if i > 100: break
			i += 1
		#if i > 100: break
		
	#compute error
	#print("dtheta size: " + str(dtheta.shape))
	#print(dtheta)
	numerator = np.linalg.norm(dtheta - dtheta_approx)    
	print("numerator: " +str(numerator))                                      
	denominator = np.linalg.norm(dtheta) + np.linalg.norm(dtheta_approx) 
	print("denominator: " +str(denominator))                                        
	difference = numerator / denominator
	
	if difference > 2e-7:
        	print ("There is a mistake in the backward propagation! difference = " + str(difference) )
    	else:
        	print ("Your backward propagation works perfectly fine! difference = " + str(difference) )
    
	
	return None

def backward_prop(Y, params, cache, activations):
	L = len(activations)
	m = Y.shape[1]
	AL = cache["A" + str(L)]
	dA_prev = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	grads = {}
	
	# Iterate down inner layers to compute grads starting from outer layer
	for l in range(L, 0, -1):
		print("l", l)
		dA = dA_prev
		A_prev = cache["A" + str(l-1)]
		Zl = cache["Z" + str(l)] 
		print("Zl", Zl)
		Wl = params["W" + str(l)]
		
		# compute Z gradient for layer
		if activations[l-1] == "sigmoid":
			g_prime = sigmoid(Zl)*(1-sigmoid(Zl))
			dZl = dA * g_prime
		if activations[l-1] == "softmax":
			dZl = cache["A" + str(l)] - Y
		if activations[l-1] == "ReLU":
			g = relu(Zl)
			g_prime = np.where(g > 0, 1, 0)
			dZl = dA * g_prime
		if activations[l-1] == "tanh":
			g_prime = 1 - tanh(Zl)**2
			dZl = dA * g_prime
			
		# compute weight/bias gradients
		dWL = np.dot(dZl, A_prev.T) / m
		dbL = np.sum(dZl, axis=1, keepdims=True) / m
		dA_prev = np.dot(Wl.T, dZl)
		
		# store gradients
		grads["dW" + str(l)] = dWL
		grads["db" + str(l)] = dbL
	
	return grads

def backward_prop_with_dropout():
    None
    return gradients


def backward_prop_with_regularization():
    None
    return gradients


def update_params(params, grads, learning_rate, optimizer="None"):

	if optimizer == "None":
        	L = len(params) // 2
        	for l in range(L):
        		params["W" + str(l+1)] -= learning_rate*grads["dW" + str(l+1)]
        		params["b" + str(l+1)] -= learning_rate*grads["db" + str(l+1)]
	elif optimizer == "momentum":
		None
	elif optimizer == "Adam":
		None
	else: raise Exception("invalid optimizer")
       
	return params


    


def model(X_test, Y_test, X_train, Y_train, learning_rate = 0.001 ,num_epochs = 1500, minibatch_size = 32, print_cost = True ):
    None
    return parameters
    
    

def predictions():
	return 




