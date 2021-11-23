import numpy as np
import math

def parameters_to_vector(weights, biases):
	assert len(weights.keys()) == len(biases.keys()), "weights and biases mismatched"
	keys = []
	vector = np.array([])
	
	#for p in list(dictionary.keys()):
	#	indexes[p] = dictionary[p].shape

	for key in list(weights.keys()):

		# flatten parameter
		weight = np.reshape(weights[key], -1)
		#print("WEGIHT SHAPE:")
		#print(weight.shape)
		bias = np.reshape(biases[key], -1)
		#print("BIASs SHAPE:")
		#print(bias.shape)
		theta = np.concatenate((weight, bias))  
		vector = np.concatenate((vector, theta))
 
	#print(keys)
	
	return vector


def vector_to_parameters(vector, orig_weights, orig_biases):
	weights = {}
	biases = {}
	i1 = 0
	i2 = 0
	for key in list(orig_weights.keys()):
		i1 = i2
		i2 = i1 + orig_weights[key].size

		weight_vector = vector[i1:i2]
		
		i1 = i2
		i2 = i1 + orig_biases[key].size

		bias_vector = vector[i1:i2]

		weights[key] = weight_vector.reshape(orig_weights[key].shape)
		
		biases[key] = bias_vector.reshape(orig_biases[key].shape)
	return weights, biases
	
	
def dictionary_to_vector(dictionary):
	"""
	Roll all our parameters dictionary into a single vector satisfying our specific required shape.
	"""
	keys = []
	count = 0
	L = len(dictionary.keys()) // 2
	shapes = {}
	
	for p in list(dictionary.keys()):
		shapes[p] = dictionary[p].shape

	for l in range(1, L+1):
        
		# flatten parameter
		new_vector = np.reshape(dictionary["W" + str(l)], (-1,1))
		keys = keys + ["W" + str(l)]*new_vector.shape[0]

		if count == 0:
			theta = new_vector
		else:
			theta = np.concatenate((theta, new_vector), axis=0)

		count = count + 1

		new_vector = np.reshape(dictionary["b" + str(l)], (-1,1))
		keys = keys + ["b" + str(l)]*new_vector.shape[0]
        
		theta = np.concatenate((theta, new_vector), axis=0)    
	#print(keys)
	
	return theta, shapes



def vector_to_dictionary(theta, shapes):
	"""
	Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
	"""
	parameters = {}
	L = len(shapes.keys()) // 2
	i1 = 0
	i2 = 0
	
	for l in range(1, L+1):
		shape_W = shapes["W"+str(l)] 
		shape_b = shapes["b"+str(l)]
		i1 = i2
		i2 = i1 + np.prod(shape_W)
		
		parameters["W"+str(l)] = theta[i1:i2].reshape(shape_W)
		i1 = i2 
		i2 = i1 + np.prod(shape_b)
		
		parameters["b"+str(l)] = theta[i1:i2].reshape(shape_b)

	return parameters

def gradients_to_vector(gradients):
	"""
	Roll all our parameters dictionary into a single vector satisfying our specific required shape.
	"""
	keys = []
	count = 0
	L = len(gradients.keys()) // 2
	shapes = {}
	
	for p in list(gradients.keys()):
		shapes[p] = gradients[p].shape

	for l in range(1, L+1):
        
		# flatten parameter
		new_vector = np.reshape(gradients["dW" + str(l)], (-1,1))
		keys = keys + ["W" + str(l)]*new_vector.shape[0]

		if count == 0:
			theta = new_vector
		else:
			theta = np.concatenate((theta, new_vector), axis=0)

		count = count + 1

		new_vector = np.reshape(gradients["db" + str(l)], (-1,1))
		keys = keys + ["b" + str(l)]*new_vector.shape[0]
        
		theta = np.concatenate((theta, new_vector), axis=0)    
	#print(keys)
	
	return theta, shapes

