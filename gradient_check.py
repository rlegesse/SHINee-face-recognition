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
height = 10
width = 10
classification = "binary"


# LOAD DATASET
print("\n \n Loading dataset...")
X_train, Y_train, X_test, Y_test = f.load_datasets(dataset, height, width, ratio)
print("Finished loading dataset! ")

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

# Normalize dataset
X_train = X_train / float(255)
X_test = X_test / float(255)

# Define model parameters / hyperparameters and set up the network
layer_dims = [X_train.shape[0], 2, C] # number of hidden units in each layer
activations = [ "ReLU", "sigmoid"]
learning_rate = 5
minibatch_size = None

L = len(layer_dims) - 1 # total number of layers

#initialize params dictionary
initialization = "Xavier"

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


#print("implementing forward propagation...")
cache, cost = f.forward_prop(params, X_train, Y_train, layer_dims, activations)
	
#print("implementing backward propagation...")
grads = f.backward_prop(Y_train, params, cache, activations)
	
#print("implementing gradient check...")
f.gradient_check(grads, params, X_train, Y_train, layer_dims, activations, epsilon = 1e-7)
	
#if epoch % 100 == 0:
print("Cost: " + str(cost))

