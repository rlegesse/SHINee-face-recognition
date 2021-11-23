# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os

def dhash(image, hashSize=8):
	# convert the image to grayscale and resize the grayscale image,
	# adding a single column (width) so we can compute the horizontal
	# gradient
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (hashSize + 1, hashSize))
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	# convert the difference image to a hash and return it
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def hamming_distance(hash1, hash2):
	# convert hashes to strings and pad with zeros to make them the same length
        # hashes are 64-bit --> 2^63 ~ 20 digits
        a = str(hash1)
	a = a.rjust(20,"0")
	b = str(hash2)
	b = b.rjust(20,"0")

	#print("Hash 1: {hash_1} Hash 2: {hash_2}".format(hash_1=a, hash_2=b))
	distance = 0

	# check that hashes are same length
	assert len(a) == len(b)

	#compare each element in strings one by one
	for i in range(len(a)):
		if a[i] != b[i]:
			distance += 1
	return distance

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-r", "--remove", type=int, default=-1,
	help="whether or not duplicates should be removed (i.e., dry run)")
ap.add_argument("-m", "--hamming-distance", type=int, default=0, help="minimum hamming distance for two images to be considered the same." )
args = vars(ap.parse_args())

# grab the paths to all images in our input dataset directory and
# then initialize our hashes dictionary
print("[INFO] computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))
hashes = {}
i = 0

# loop over our image paths
for imagePath in imagePaths:
	
	# load the input image and compute the hash
	image = cv2.imread(imagePath)
	h = dhash(image)
	#print("current image: ", imagePath)
	
	if i == 0: # if this is the first iteration
		p = []
		p.append(imagePath)
		hashes[h] = p
		#print("\n first hash: {hash1}  path: {path1}".format(hash1=h, path1=p))
		i +=1
	elif i > 0: 
		unique = False
		#iterate over existing hashes in dictionary
		for hash_ in list(hashes):
			# compare current image's hash to hash in dictionary
			if hash_ != None:
				distance = hamming_distance(hash_, h)
				#print("comparing {image1} with {image2}".format(image1 = imagePath, image2=hashes[hash_]))
				# if images are similar, add image path to hashes' list
				if distance <= args["hamming_distance"]:
					print("similar!")
					p = hashes.get(hash_, [])
					p.append(imagePath)
					hashes[hash_] = p
					#print("hash for: {name}... appending current path: {path2} \n".format(name=hashes[hash_], path2=p))
					unique = False
					break
				else: # save current image hash as a new entry in dictionary
					unique = True
			else: print("No hash")
                # current image matches with no existing entry
                if unique == True :
                    p = []
                    p.append(imagePath)
                    hashes[h] = p
                    #print("new hash entry \n")
                            
			
i = 0

# loop over the image hashes
for (h, hashedPaths) in hashes.items():
	
	#print(hashedPaths)
        # check to see if there is more than one image with the same hash
	if len(hashedPaths) > 1:
		print("there is more than one image with this hash")
		
		# initialize a montage to store all images with the same hash
		montage = None
		
		# loop over all image paths with the same hash
		for p in hashedPaths:
			# load the input image and resize it to a fixed width
			# and heightG
			image = cv2.imread(p)
			image = cv2.resize(image, (150, 150))
			# if our montage is None, initialize it
			if montage is None:
				montage = image
			# otherwise, horizontally stack the images
			else:
				montage = np.hstack([montage, image])
				# show the montage for the hash
				#print("[INFO] hash: {}".format(h))
				cv2.imshow("Montage", montage)
				
				# check to see if this is a dry run
				if args["remove"] <= 0:
					cv2.waitKey(0)
				
				else:
					print("To delete the image, press 'd'. To keep, press 'k'")
					wait = True
					while wait:
						if cv2.waitKey(0) == ord('d'):
							os.rename(p,"./duplicates/" + str(i))
							i+=1
							print("deleted {p}".format(p=p))
							wait = False
						elif cv2.waitKey(0) == ord('k'):
							wait = False
							print("skipped")
						else:
							print("invalid key!")
					

