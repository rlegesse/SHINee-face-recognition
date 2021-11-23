import argparse
import matplotlib.pyplot as plt

import nn_utils as nn
import vector_ops as vec
import data_ops as data
import models as model


# define arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

directory = args["dataset"]
img_height = 128
img_width = 128

#******************************************************
# 		LOAD DATASET (MINI BATCHES)		*
#******************************************************


# put lists of data files into dictionary
# dataset must be organized in folders corresponding to label

dataset, label_map = data.directory_to_dictionary(directory)
for n,l in label_map: print("Class '{n}' has label {l}".format(n=n, l=l)) 
num_classes = len(dataset)


#shuffle order of file names
data.shuffle_dictionary(dataset)


#balance classes in dataset
data.balance_data(dataset)


#split dataset into train and test sets
train_set, test_set = data.split_dictionary(dataset, 0.7)


#Note: to maintain class balance, minibatches contain equal proportions
#of data from each class. 
#as class with least amount of data. Therefore some data will not be used
#minibatches = data.dictionary_to_minibatches(train_set, batch_size=8)
minibatches = data.random_minibatches(train_set, batch_size=6)

print("\nTrain set size: {s}".format(s=data.dataset_size(train_set)))
print("Test set size: {s}".format(s=data.dataset_size(test_set)))


#generate model
model = model.Model(img_height*img_width)
model.add_layer(40, "relu")
model.add_layer(20, "tanh")
model.add_layer(15, "relu")
model.add_layer(15, "sigmoid")
model.add_layer(10, "relu")
model.add_layer(10, "sigmoid")
model.add_layer(10, "relu")
model.add_layer(10, "sigmoid")
model.add_layer(num_classes, "softmax")
model.summary()

#initialize parameters
model.initialize("Xavier")
accuracy = 0
epochs = 100

for epoch in range(epochs):
	
	#shuffle train set
	data.shuffle_dictionary(train_set)

	#split train set into minibatches
	minibatches = data.balanced_minibatches(train_set, batch_size=8)

	# train over mini-batches
	for minibatch in minibatches: 
		
		#load minibatch to array
		minibatch_X, minibatch_Y = data.load_batch_to_array(minibatch, img_height, img_width)
		
		#flatten array to input vector
		minibatch_X = data.flatten_image_array(minibatch_X)
		
		#normalize input vector 
		minibatch_X = minibatch_X / float(255)
		
		#convert labels to one-hot vector
		minibatch_Y = data.one_hot_encoder(minibatch_Y, num_classes)
		
		#shuffle data
		minibatch_X, minibatch_Y = data.shuffle_data(minibatch_X, minibatch_Y)
		
		#run forward/back prop once per minibatch
		model.fit(minibatch_X, minibatch_Y, 1, 0.05)
		
		#get accuracy for each minibatch on last epoch
		if epoch == epochs-1:
			accuracy += model.accuracy()
			print(model.accuracy())
			
	print("Cost after {i} epochs: {c}".format(i = epoch, c = model.cost()))
	
accuracy = accuracy / len(minibatches)	
print("train accuracy: {a}".format(a = accuracy))

#TODO compute test accuracy
#divide test samples to batches (to save memory)
#load, flatten, and normalize images
# for each batch, run forward pass (make method called test that prints accuracy and shows predictions??)

model.plot_costs()

#model.predict(X_test, Y_test)
#model.gradient_check(X_train, Y_train)


#load test data array
#X_test, Y_test, label_map = data.load_files_to_array(test_set, 100, 100)


# PREPROCESSING
# Encode labels into one-hot matrices
#Y_test = data.one_hot_encoder(Y_test, C)

# Normalize dataset
#X_test = X_test / float(255)


