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
		self.__gradients = [{}]*2
		self.__cache = None
		self.__activations = [" "]  
		self.__dimensions = [input_size]
		self.__is_initialized = False
		self.__logits = None
		self.__cost = None


	def add_layer(self, units, activation):
		self.__activations.append(activation)
		self.__dimensions.append(units)
		
	
	def initialize(self, initialization="random"):		
		initializer = Initializer(initialization)
		initializer(self.__dimensions, self.__weights, self.__biases)
		self.__is_initialized = True
		
		 
	def __forward_pass(self, _input, optimization):
		self.__cache, self.__logits = nn.forward_pass(_input, 
							   self.__activations, 
							   self.__weights,
							   self.__biases)
	
	
	def __backward_pass(self, labels):
		batch_size = labels.shape[1] 
		dA = nn.cross_entropy_derivative(labels, self.logits)
		
		for layer, activation in reversed(list(enumerate(self.__activations, start=1))):
			Z = self.__cache[0][layer]
			A = self.__cache[1][layer]
			dZ, dA, dW, db = nn.backward_step(Z, dA, self.__weights[layer], A, activation)
			self.__gradients[0][layer] = dW
			self.__gradients[1][layer] = db	
			
			
	def __update_parameters(self, learning_rate):
		for layer in range(1, len(self.__dimensions)):
			self.__weights[layer] -= self.__gradients[0]*learning_rate
			self.__biases[layer] -= self.__gradients[1]*learning_rate 
	
		
	def fit(self, _input, labels, epochs=100, learning_rate=0.1, optimization=None):
		assert self.__is_initiazed == True, "Must call initialize() first!"
		for epoch in range(epochs):
			self.__forward_pass(_input, optimization)
			cost = nn.compute_cost(self.__logits, labels)
			self.__costs.append(cost)
			self.__backward_pass(labels)
			self.__update_parameters(learning_rate)
			if epoch % 100 == 0: 
				print("Cost after {epoch} epochs: {cost}".format(epoch = epoch, cost = cost))
			 
	
	def plot_costs():
		plt.plot(self.__costs)
		plt.show()	
		
		
	def predict(inputs, labels): 
		self.__forward_pass(inputs, None)
		
		return predictions
	
	
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
			weights[l] = np.zeros(dimensions[l], dimensions[l-1]) * 0.01 # scale to prevent explosion
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

