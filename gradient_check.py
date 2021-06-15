import cv2
import numpy as np
from numpy import asarray
from imutils import paths
import argparse
import os
import math

import matplotlib.pyplot as plt
from matplotlib import cm
#import h5py
import scipy
from PIL import Image
from scipy import ndimage
import functions as f



# define arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-r", "--ratio", required=True, help="ratio of training examples to test examples (number from 0 to 1)")
args = vars(ap.parse_args())

# load test data from dataset folder
dataset = args["dataset"]
ratio = float(args["ratio"])
height = 100
width = 100
classification = "multi class"


# LOAD DATASET
print("\n \n Loading dataset...")
X_train, Y_train, X_test, Y_test = f.load_datasets(dataset, height, width, ratio)
print("Finished loading dataset! ")

print(X_train.shape)
print(Y_train.shape)

m = X_train.shape[0]
C = len(os.listdir(dataset)) #number of classes



# PREPROCESSING


# Encode labels into one-hot matrices for muli-class classification
# or 1xm vector 0s and 1s for binary classification
if classification == "multi class": 
	Y_train = f.one_hot_encoder(Y_train, C)
	Y_test = f.one_hot_encoder(Y_test, C)
	print("number of classes " + str(C))
if classification == "binary":
	Y_train = Y_train.T - 1
	C = 1
	print("binary classification")


# Flatten dataset
X_train, X_test = f.flatten_dataset(X_train, X_test)

print(X_train.shape)
print(Y_train.shape)

# Shuffle dataset
X_train, Y_train = f.shuffle_data(X_train, Y_train)

print(Y_train)

# Normalize dataset
X_train = X_train / float(255)
X_test = X_test / float(255)

# Define model parameters / hyperparameters and set up the network
layer_dims = [X_train.shape[0], 5, 5, C] # number of hidden units in each layer
activations = [ "tanh", "tanh", "softmax"]
learning_rate = 0.8
minibatch_size = None

L = len(layer_dims) - 1 # total number of layers

#initialize params dictionary
initialization = "He"

#initialize param values
if initialization == "Xavier":
	params = f.initialize_params_Xavier(layer_dims)
elif initialization == "random": 
	params = f.initialize_params(layer_dims)
elif initialization == "He":
	params = f.initialize_params_He(layer_dims)
else: raise Exception("Invalid initialization")

	
#print(params)
#print(Y_train)
costs = np.array([[]])

for epoch in range(1):
	#print("implementing forward propagation...")
	cache, A = f.forward_prop(params, X_train, Y_train, L, activations)			
	cost = f.compute_softmax_cost(A, Y_train)
	costs = np.append(costs, cost)
	
	#print("implementing backward propagation...")
	grads = f.backward_prop(Y_train, params, cache, activations)
	
	if epoch % 100 == 0:
		print("Cost after {e} epochs: {c}".format(e=epoch, c=cost))
	if epoch == 999: break
	params = f.update_params(params, grads, learning_rate, optimizer="None")
	

print("implementing gradient check...")
f.gradient_check_2(params, grads, X_train, Y_train, L, activations, epsilon = 1e-7)
		 
plt.plot(costs)
plt.show()


