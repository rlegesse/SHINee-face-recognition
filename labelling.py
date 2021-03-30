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
#from lr_utils import load_dataset



def load_datasets(root, height, width, train_ratio=0.7):
	
	if root[len(root)-1] != "/" : root = root + "/"
	dirs = os.listdir(root)
	dirs = [root + i for i in dirs]
	
	dataset_size = len(list(paths.list_images(root)))
	print("dataset size = " ,type(dataset_size))
	
	trainset_size = int(dataset_size*train_ratio)
	print("training set size = " + str(trainset_size))
	
	testset_size = dataset_size - trainset_size
	print("training set size = " + str(testset_size))
	
	X_train = np.ndarray(shape=(trainset_size, height, width, 3), dtype=np.uint8) 
	Y_train = np.ndarray(shape=(trainset_size, 1), dtype=int)
	X_test = np.ndarray(shape=(testset_size, height, width, 3), dtype=np.uint8)
	Y_test = np.ndarray(shape=(testset_size, 1), dtype=int)
	
	label = 0
	i = 0
	k = 0 
	
	for subset in dirs:	
		print("This is subset '{subset}'. It has label {label}".format(subset=subset, label=label))
		imagePaths = list(paths.list_images(subset))
		m = len(imagePaths)
		n_train = int(m*float(train_ratio))
		
		for j in range(m): 
			
			# Get n training examples and reshape	
			img = np.array(ndimage.imread(imagePaths[j], flatten=False))	
			
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







	
	
	
	
	
	
	
	
	
	
	
