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

import nn_utils as nn
import vector_ops as vec
import data_ops as data
import models as model

# define arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-r", "--ratio", required=True, help="ratio of training examples to test examples (number from 0 to 1)")
args = vars(ap.parse_args())

# load test data from dataset folder
dataset = args["dataset"]
ratio = float(args["ratio"])
height = 256
width = 256
classification = "multi class"


# LOAD DATASET
print("\n \n Loading dataset...")
X_train, Y_train, X_test, Y_test = data.load_datasets(dataset, height, width, ratio)
print("Finished loading dataset! ")

m = X_train.shape[0]
C = len(os.listdir(dataset)) #number of classes

# PREPROCESSING
# Encode labels into one-hot matrices for muli-class classification
# or 1xm vector 0s and 1s for binary classification

#if classification == "multi class": 
Y_train = data.one_hot_encoder(Y_train, C)
Y_test = data.one_hot_encoder(Y_test, C)

#if classification == "binary":
#	Y_train = Y_train.T - 1
#	C = 1

print("number of classes " + str(C))

# Flatten dataset
X_train, X_test = data.flatten_dataset(X_train, X_test)

# Shuffle dataset
X_train, Y_train = data.shuffle_data(X_train, Y_train)

# Normalize dataset
X_train = X_train / float(255)
X_test = X_test / float(255)

#generate model
model = model.Model(X_train.shape[0])

model.add_layer(10, "relu")
model.add_layer(20, "relu")
model.add_layer(20, "sigmoid")
model.add_layer(C, "softmax")
model.summary()

model.initialize("Xavier")

model.fit(X_train, Y_train, 100, 0.1, None)
model.plot_costs()



