import numpy as np
from numpy import asarray

import math
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
from PIL import Image
from scipy import ndimage
from decimal import Decimal

import vector_ops as vec
import decimal

#******************************************************
#*          LINEAR & ACTIVATION FUNCTIONS             *
#******************************************************

def linear_function(W, x, b):
	z = np.dot(W, x) + b
	return z
	
	
def sigmoid(Z):
	A = 1 / (1 + np.exp(-Z))
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



#******************************************************
#*             ACTIVATION DERIVATIVES                 *
#******************************************************

def sigmoid_derivative(_input):
	derivative = sigmoid(_input)*(1-sigmoid(_input)) + 1e-7
	return derivative       
	
	
def relu_derivative(_input): 
	derivative = np.where(relu(_input) > 0, 1, 0)
	return derivative        
	
	
def tanh_derivative(_input):
	derivative = 1 - tanh(_input)**2
	return derivative         
	
	
def softmax_jacobian(Z): # returns a tensor of jacobian matrices for each example 
	a = softmax(Z)
	m = a.shape[1]
	n = a.shape[0]
	DM = (a.T).reshape(m,-1,1) * np.identity(n) #diagonal matrix
	OP = (a.T).reshape(m,-1,1) * (a.T).reshape(m,1,-1) #outer product 
	jacobian = DM - OP
	return jacobian

	
	

#******************************************************
#*               FORWARD PROPAGATION                  *
#******************************************************


def activate(_input, activation):
	activations = {
		"relu": relu,
		"sigmoid": sigmoid,
		"tanh": tanh,
		"softmax":softmax
	}
	output = activations[activation](_input)
	return output

    
def forward_pass(input_x, activations, weights, biases):
	linear_cache = {}
	activation_cache = {}
	linear_input = input_x
	activation_cache[0] = input_x
	for layer, activation in enumerate(activations[1:], start=1):
		if layer > 0:
			linear_output = linear_function(weights[layer], linear_input, biases[layer]) 
			activation_output = activate(linear_output, activation)
			
			linear_cache[layer] = linear_output
			activation_cache[layer] = activation_output
			
			linear_input = activation_output
		
	return activation_output, linear_cache, activation_cache
    
    
#******************************************************
#*                   COST FUNCTIONS                   *
#******************************************************


def compute_cost(logits, labels):
	if labels.shape[0] == 1:
		cost = binary_cross_entropy(logits, labels)
	else: cost = categorical_cross_entropy(logits, labels)
	return cost
	
	
def binary_cross_entropy(A, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A),Y) + np.multiply(np.log(1-A),1-Y)
    cost = -np.sum(logprobs)/m
    return float(np.squeeze(cost))


def categorical_cross_entropy(A, Y):
	m = Y.shape[1]
	#A = A.astype(decimal.Decimal)
	#log_A = np.zeros(A.shape, dtype=decimal.Decimal) 
	#loss = Decimal(0)
	#print(A, Y)
	#for i in range(A.shape[0]):
	#	for j in range(A.shape[1]):
	#		loss += Decimal(Y[i][j]) * Decimal(math.log(Decimal(A[i][j])))
	np.set_printoptions(precision=3)
	
	L = np.sum(np.log(A+1e-7)*Y,  axis=0)
	#return -Decimal(1./m)*loss  
    	return -(1./m)*np.sum(L)
    
     
#******************************************************
#*               BACKWARD PROPAGATION                 *
#******************************************************

def cross_entropy_derivative(logits, labels):
	if labels.shape[0] == 1:
		derivative = binary_loss_derivative(logits, labels)
	else: derivative = categorical_loss_derivative(logits, labels)
	return derivative
	

def binary_loss_derivative(logits, labels):
	derivative = np.divide(labels, logits) + np.divide((1-labels), (1-logits))
	return derivative
	
	
def categorical_loss_derivative(logits, labels):
	batch_size = labels.shape[1]
	derivative = -(1.0/batch_size)*np.divide(labels, logits)
	return derivative


def compute_gradients(linear_grad, activation_value):
	batch_size = activation_value.shape[1]
	weight_grad = np.dot(linear_grad, activation_value.T)   #dW = dot(dZ, A_prev) / m
	bias_grad = np.sum(linear_grad, axis=1, keepdims=True)   #db = sum(dZ) / m
	return weight_grad, bias_grad


def activation_derivative(input_Z, dA, activation):
	activations = {
		"relu": relu_derivative,
		"sigmoid": sigmoid_derivative,
		"tanh": tanh_derivative,
		"softmax": softmax_jacobian
	}
	batch_size = input_Z.shape[1]
	g_prime = activations[activation](input_Z)
	#print("\ndA")
	#print(dA.shape)
	#print("\ng_prime")
	#print(g_prime.shape)
	
	if activation == "softmax":
		dZ = np.array([np.dot(dA[:,i], g_prime[i]) for i in range(batch_size)]).T
	else: dZ = dA * g_prime
	
	return dZ
	
	
def backward_pass(logits, labels, activations, weights, linear_cache, activation_cache):
	weight_grads = {}
	bias_grads = {}
	dA = cross_entropy_derivative(logits, labels)
	for layer, activation in reversed(list(enumerate(activations[1:], start=1))):
		Z = linear_cache[layer]
		A_prev = activation_cache[layer-1]
		
		dZ = activation_derivative(Z, dA, activation)
		dA = np.dot(weights[layer].T, dZ)
		
		dW, db = compute_gradients(dZ, A_prev)
		weight_grads[layer] = dW
		bias_grads[layer] = db	
	
	return weight_grads, bias_grads

"""def back_prop(labels, weights, linear_cache, activation_cache, activations):
	m = labels.shape[1]
	weight_grads = {}
	bias_grads = {}
	
	layers = activations[1:]
	L = len(layers)
	AL = activation_cache[L]
	dA = -(1.0/m)*np.divide(labels, AL).T
	
	# Iterate down inner layers to compute grads starting from outer layer
	for l in range(L, 0, -1):
		
		# compute dZ for current layer
		Zl = linear_cache[l] 
		#print("Current layer: " + str(l))
		
		if layers[l-1] == "softmax":
			g_prime = softmax_jacobian(Zl)
			#g_prime = (1.0/m)*(AL - Y)
			dZl = np.array([np.dot(dA[i], g_prime[i]) for i in range(m)]).T  # tensor-matrix multiplication
	
		if layers[l-1] == "sigmoid":
			g_prime = sigmoid(Zl)*(1-sigmoid(Zl))
			dZl = dA * g_prime
		
		if layers[l-1] == "ReLU":
			g = relu(Zl)
			g_prime = np.where(g > 0, 1, 0)
			dZl = dA * g_prime
			
		if layers[l-1] == "tanh":
			g_prime = 1 - tanh(Zl)**2
			dZl = dA * g_prime
			
		# compute weight/bias gradients
		Wl = weights[l]
		A_prev = activation_cache[l-1]
		dWl = np.dot(dZl, A_prev.T) / m
		#print("dWl shape: " + str(dWl.shape))
		dbl = np.sum(dZl, axis=1, keepdims=True) / m
		
		# compute dA for preceding layer
		dA = np.dot(Wl.T, dZl)
		
		# store gradients
		weight_grads[l] = dWl
		bias_grads[l] = dbl
	return weight_grads, bias_grads
"""	

#******************************************************
#*                   GRADIENT CHECK                   *
#******************************************************

def gradient_check(input_x, labels, weight_grads, bias_grads, activations, weights, biases, epsilon=1e-5):
	theta = vec.parameters_to_vector(weights, biases)
	dtheta = vec.parameters_to_vector(weight_grads, bias_grads)
	dtheta_approx = np.zeros(dtheta.shape)

	print("function gradient:\t\tNumerical gradient:")
	for i in range(theta.size):
		theta_plus = np.copy(theta)
		theta_plus[i] += epsilon

		theta_minus = np.copy(theta)
		theta_minus[i] -= epsilon
		
		new_weights, new_biases = vec.vector_to_parameters(theta_plus, weights, biases)
		logits, _, _ = forward_pass(input_x, activations, new_weights, new_biases)
		J_plus = compute_cost(logits, labels)
		
		new_weights, new_biases = vec.vector_to_parameters(theta_minus, weights, biases)
		logits, _, _ = forward_pass(input_x, activations, new_weights, new_biases)
		J_minus = compute_cost(logits, labels)
		
		#dtheta_approx[i] = (J_plus - J_minus) / Decimal(2.*epsilon)
		dtheta_approx[i] = (J_plus - J_minus) / (2.*epsilon)
		

		#print("{f}\t\t{n}".format(n=dtheta_approx[i], f=dtheta[i])) 
		#print((dtheta_approx[i] - dtheta[i]) / dtheta_approx[i])
	numerator = np.linalg.norm(dtheta_approx - dtheta)
	denominator = np.linalg.norm(dtheta_approx) + np.linalg.norm(dtheta)
	error = numerator / denominator
	
	if error > 2e-7:
		print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(error) + "\033[0m")
    	else:
    		print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(error) + "\033[0m")
	
	return


#******************************************************
#*                   ACCURACY                   *
#******************************************************

def top1_accuracy(a, labels):
	m = labels.shape[1]
	correct = 0.
	for i in range(m):
		if np.argmax(labels[:,[i]]) == np.argmax(a[:,[i]]):
			correct += 1
	return correct / m


