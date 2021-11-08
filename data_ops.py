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
	

def shuffle_data(X, Y):
	
	shuffler = np.random.permutation(X.shape[1])
	print(X.shape[1])
	X_shuffled = X.T[shuffler]
	Y_shuffled = Y.T[shuffler]
	
	return X_shuffled.T, Y_shuffled.T	
	
