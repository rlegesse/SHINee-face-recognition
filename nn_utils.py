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
from decimal import Decimal

import vector_ops as vec


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
#*         ACTIVATION DERIVATIVES            *
#******************************************************

def sigmoid_derivative(_input):
	derivative = sigmoid(_input)*(1-sigmoid(_input))
	return derivative       
	
	
def relu_derivative(): 
	derivative = np.where(relu(Zl) > 0, 1, 0)
	return derivative        
	
	
def tanh_derivative():
	derivative = 1 - tanh(Zl)**2
	return derivative         
	
	
def softmax_jacobian(Z): # returns a tensor of jacobian matrices for each example 
	a = softmax(Z)
	m = a.shape[1]
	n = a.shape[0]
	DM = (a.T).reshape(m,-1,1) * np.identity(n) #diagonal matrix
	OP = (a.T).reshape(m,-1,1) * (a.T).reshape(m,1,-1) #outer product 
	jacobian = DM - OP
	return jacobian


def softmax_derivative(_input):
	g_prime = softmax_jacobian(_input)
	linear_grad = np.array([np.dot(dA[i], g_prime[i]) for i in range(m)]).T       # tensor-matrix multiplication
	return linear_grad                                 #dZ = dA * g'

	
	

#******************************************************
#*               FORWARD PROPAGATION                  *
#******************************************************


def activate(_input, activation):
	activations = {
		"relu": relu,
		"sigmoid": sigmoid,
		"tanh": tanh
	}
	output = activations[activation](_input)
	return output


def forward_step(_input, weight, bias, activation):
	linear_output = linear_function(weight, _input, bias)
	activation_output = activate(linear_output, activation)
	return acivation_output, linear_output


def forward_pass(_input, activations, weights, biases):	
	linear_input = _input	
	for layer, activation in enumerate(activations, start=1):	
		 weight = weights[layer]
		 bias = biases[layer]
		 activation_output, linear_output = forward_step(linear_input, 
		 							  weight, 
		 							  bias, 
		 							  activation) 
		 activation_cache[layer] = activation_output
		 linear_cache[layer] = linear_output
		 linear_input = linear_output	
	
	logits = activation_output
		 	 
	return (activation_cache, linear_cache), logits
 
    
    
    
#******************************************************
#*                   COST FUNCTIONS                   *
#******************************************************


def compute_cost(logits, labels):
	if Y.shape[0] == 1:
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
	L = np.sum(np.log(A) * Y, axis=0)
	return -(1./m)*np.sum(L)
	
#TODO:	
def compute_cost_with_regularization():
    None
    return   
    
    
     
#******************************************************
#*               BACKWARD PROPAGATION                 *
#******************************************************

def cross_entropy_derivative(logits, labels):
	if labels.shape[0] == 1:
		derivative = binary_loss_derivative(logits, labels)
	else: derivative = categorical_loss_derivative(logits, labels)
	return derivative
	

def binary_loss_derivative(logits, labels):
	derivative = -(1.0/m)*np.divide(labels, logits).T
	return derivative
	
	
def categorical_loss_derivative():
	derivative = np.divide(labels, logits) + np.divide((1-labels), (1-logits))
	return derivative


def compute_gradients(linear_grad, activation_value, batch_size):
	weight_grad = np.dot(linear_grad, activation_value.T) / batch_size  #dW = dot(dZ, A_prev) / m
	bias_grad = np.sum(linear_grad, axis=1, keepdims=True) / batch_size  #db = sum(dZ) / m
	return weight_grad, bias_grad


def activation_derivative(input_Z, dA, activation):
	activations = {
		"relu": relu_derivative,
		"sigmoid": sigmoid_sigmoid_derivative,
		"tanh": tanh_derivative,
		"softmax": softmax_derivative
	}
	g_prime = activations[activation](input_Z)
	if activation == "softmax":
		dZ = np.array([np.dot(dA[i], g_prime[i]) for i in range(m)]).T
	else: dZ = dA * g_prime
	return dZ
	
	
def backward_step(input_Z, dA, weight, A, activation):
	dZ = activation_derivative(input_Z, dA, activation)
	dW, db = compute_gradients(dZ, A, batch_size)
	dA = np.dot(weight.T, dZl)
	return dZ, dA, dW, db


	

