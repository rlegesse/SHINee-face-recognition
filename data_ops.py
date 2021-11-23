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
import random

  
#******************************************************
#		 ARRAY OPERATIONS   	        	*
#******************************************************

def flatten_image_array(array):
	samples, width, height = array.shape
	flat_vector = array.reshape(height*width, samples)
	return flat_vector
	
	
def one_hot_encoder(Y_orig, classes):
	Y = np.zeros((classes, len(Y_orig)))
	for i in range(len(Y_orig)):
		Y[Y_orig[i]-1,i] = 1
	return Y
	

def shuffle_data(X, Y):
	
	shuffler = np.random.permutation(X.shape[1])
	#print(X.shape[1])
	X_shuffled = X.T[shuffler]
	Y_shuffled = Y.T[shuffler]
	
	return X_shuffled.T, Y_shuffled.T	
	
	
#******************************************************
#		 DATASET OPERATIONS	        	*
#******************************************************
	
def directory_to_dictionary(root):
	#returns dictionary mapping each class folder to a list of its image files
	if root[len(root)-1] != "/" : root = root + "/"
	dirs = os.listdir(root)	
	file_dictionary = {}
	label = 0
	label_map = []
	for folder in dirs:
		file_dictionary[label] = list(paths.list_images(root+folder))
		label_map.append((folder, label))
		label +=1
	return file_dictionary, label_map


def balance_data(dictionary):
	# equalize number of samples from each class (truncate extra data)
	min_class_size = 100000000
	
	# find class with lowest number of samples
	for subset, file_list in dictionary.items():
		if min_class_size > len(file_list):
			min_class_size = len(file_list)
	
	#truncate data exceeding minimum	
	for subset, file_list in dictionary.items(): 
		del file_list[min_class_size:]
	
	return
	
	
def dataset_size(dataset):
	size = 0
	for key, file_names in dataset.items():
		size += len(file_names)
	return size
	
	
def shuffle_dictionary(dictionary):
	for key in dictionary.keys():
		random.shuffle(dictionary[key])
	return
	
	
def split_dictionary(dictionary, split_ratio=0.7):
	train_dictionary = {}
	test_dictionary = {}
	
	for key in dictionary.keys():
		train_size = int(round(len(dictionary[key])*split_ratio))
		train_dictionary[key] = dictionary[key][0:train_size]
		test_dictionary[key] = dictionary[key][train_size:]
		
	return train_dictionary, test_dictionary


def random_minibatches(dictionary, batch_size):
	#generates list of random minibatches -> list of tuples (label, img_file) 
	
	# convert data dictionary to list of tuples (label, file name)
	data = []
	for label, filelist in dictionary.items():
		data += [ (label, img) for img in filelist ]
	
	random.shuffle(data)
	
	num_minibatches = int(len(data) / batch_size)
	minibatches = []
	t = 0
	
	for i in range(num_minibatches):
		minibatches.append(data[t:t+batch_size])
		t += batch_size
	
	return minibatches


def balanced_minibatches(dictionary, batch_size=32):
	#generates list of balanced minibatches -> list of tuples (label, img_file) 

	assert batch_size % len(dictionary) == 0, "Batch size must be a multiple of the number of classes for class balance!"
	
	balance_data(dictionary)
	
	samples_per_class = int(batch_size / len(dictionary))	
	number_of_minibatches = len(dictionary[0]) / samples_per_class

	minibatches = []
	t = 0
	
	for i in range(number_of_minibatches):
		minibatch = []
		for label, filelist in dictionary.items():
			minibatch +=  [ (label, img) for img in filelist[t:t + samples_per_class] ]
		minibatches.append(minibatch)
		t += samples_per_class
	
	return minibatches
	
	
def load_batch_to_array(batch, img_height, img_width):
	# takes list of tuples (label, image_file) 
	# and loads labels and image data into arrays X and Y 
	
	batch_size = len(batch)
	X_array = np.ndarray(shape=(batch_size, img_height, img_width), dtype=np.uint8) 
	Y_array = np.ndarray(shape=(batch_size, 1), dtype=int)
	
	i = 0
	for (label, filename) in batch:
		image = np.array(ndimage.imread(filename, flatten=True))
			
		if image.shape != (img_height, img_width, 3): 
			image = cv2.resize(image, (img_height, img_width))
			
		X_array[i] = image
		Y_array[i] = label
		i += 1
		
	return X_array, Y_array

	
