import numpy as np
import math
import cv2
from numpy import asarray
from imutils import paths
import os
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy
from PIL import Image
from scipy import ndimage
from decimal import Decimal

import nn_utils as nn
import vector_ops as vec
import data_ops as data


class Model:
	def __init__(self, input_size):
		self.__weights = {}
		self.__biases = {}
		self.__weight_grads = {}
		self.__bias_grads = {}
		self.__activation_cache = {}
		self.__linear_cache = {}
		self.__activations = [" "]  
		self.__dimensions = [input_size]
		self.__is_initialized = False
		self.__logits = None
		self.__cost = None
		self.__costs = []
		self.__trained = False


	def add_layer(self, units, activation):
		self.__activations.append(activation)
		self.__dimensions.append(units)
		
	
	def initialize(self, initialization="random"):		
		initializer = Initializer(initialization)
		initializer(self.__dimensions, self.__weights, self.__biases)
		self.__is_initialized = True
		
		 
	def __forward_pass(self, _input):
		self.__logits, self.__linear_cache, self.__activation_cache = nn.forward_pass(_input, 
								self.__activations, 
								self.__weights,
								self.__biases)
	
	def __backward_pass(self, labels):	
		self.__weight_grads, self.__bias_grads = nn.backward_pass(self.__logits, labels, self.__activations, 
								self.__weights, self.__linear_cache, self.__activation_cache)
			
			
	def __update_parameters(self, learning_rate):
		for layer in range(1, len(self.__dimensions)):
			self.__weights[layer] -= self.__weight_grads[layer]*learning_rate
			self.__biases[layer] -= self.__bias_grads[layer]*learning_rate 
	
	
	def fit(self, nn_input, labels, iterations=1, learning_rate=0.1):		
		assert self.__dimensions[-1] == labels.shape[0], "Output layer dimensions do not match number of classes."
		assert self.__is_initialized == True, "Must call initialize() first!"
		
		for iteration in range(iterations):
			self.__forward_pass(nn_input)
			self.__cost = nn.compute_cost(self.__logits, labels)
			self.__costs.append(self.__cost)
			self.__backward_pass(labels)
			self.__update_parameters(learning_rate)
		
		self.__trained = True 
		self.__accuracy = nn.top1_accuracy(self.__logits, labels)
		
		
	def cost(self):
		return(self.__cost)
	
				 
	def gradient_check(self, nn_input, labels):
		assert self.__trained == True, "train first!"
		nn.gradient_check(nn_input, labels, self.__weight_grads, self.__bias_grads, self.__activations, 
					self.__weights, self.__biases, epsilon=1e-5)
					
					
	def plot_costs(self):
		plt.plot(self.__costs)
		plt.show()	
		
		
	def predict(self, inputs, labels): 
		self.__forward_pass(inputs)
		predictions = []		
		print("test accuracy: {accuracy}".format(accuracy = self.accuracy(labels)))
		for i in range(labels.shape[1]):
			predictions.append(np.argmax(self.__logits[:,[i]]))
		return predictions 
	
	
	def accuracy(self):
		return self.__accuracy
	
	
	def summary(self):
		assert len(self.__dimensions) > 1, "Model is empty"
		print("\nMODEL SUMMARY: \nLAYER\t\tHIDDEN UNITS\tACTIVATION")
		for layer, units in enumerate(self.__dimensions):
			print(str(layer) + "\t\t" + str(units) + "\t\t" + self.__activations[layer])  
		print("\n")
		return


class Initializer:
	def __init__(self, initialization):
		self.initialization = initialization
		self.__initializers = {
			"random": self.__random_initializer,
			"He": self.__He_initializer,
			"Xavier": self.__Xavier_initializer,
			"zeros": self.__zeros_initializer
		}
		
	def __call__(self, dimensions, weights, biases):
		self.__initializers[self.initialization](weights, biases, dimensions)
		return
	
		
	def __random_initializer(self, weights, biases, dimensions):  # initialize weights (randomly) and biases (zeros)
		for l in range(1, len(dimensions)):
			weights[l] = np.random.randn(dimensions[l], dimensions[l-1]) * 0.01 # scale to prevent explosion
			biases[l] = np.zeros((dimensions[l], 1))
		return


	def __zeros_initializer(self, weights, biases, dimensions):
		for l in range(1, len(dimensions)):
			weights[l] = np.zeros(dimensions[l], dimensions[l-1]) 
			biases[l] = np.zeros((dimensions[l], 1))
		return


	def __Xavier_initializer(self, weights, biases, dimensions):
		for l in range(1, len(dimensions)):
			weights[l] = np.random.randn(dimensions[l], dimensions[l-1]) * np.sqrt(1./dimensions[l-1])
			biases[l] = np.zeros((dimensions[l], 1))
		return 
		
		
	def __He_initializer(self, weights, biases, dimensions):
		for l in range(1, len(dimensions)):
			weights[l] = np.random.randn(dimensions[l], dimensions[l-1]) * np.sqrt(2./dimensions[l-1])
			biases[l] = np.zeros((dimensions[l], 1)) 
		return 

