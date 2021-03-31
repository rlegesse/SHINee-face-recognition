# SHINee-face-recognition

## Introduction
The purpose of this project is to create a machine learning model capable of identifying the members of the [Korean boy group SHINee(샤이니)](https://youtu.be/7z62LsMQ-_g?t=98 "SHINee(샤이니)"). from image data.
I was motivated by my desire to explore machine learning and improve my development skills. What better way to do this than to make it about my favorite K-Pop group!

I am developing a deep neural network from scratch rather than using a machine learning framework (such as Keras or Tensorflow) in order to gain a deeper understanding of the inner workings of neural networks.

In addition to the model itself, this project also includes scripts for preprocessing training data. 


## Technologies
Developed on Ubuntu 20.04.2  /  LTS Kernel: Linux 5.8.0-45-generic

Using:
* Python 2.7.18
* OpenCV 4.5.1

## Launch

1. Install required packages [INSERT COMMAND HERE]
2. Run model on sample dataset "dataset/" [INSERT COMMAND HERE]

## Methodology
[INSERT OVERVIEW OF PROJECT METHODOLOGY HERE]

## Optional Data Collection
I will post a link to the full dataset used to train the model at a later point in time. If you would like to gather your own training data, follow the steps below:
1. First run [ostrolucky's Bulk-Bing-Image-Downloader](https://github.com/ostrolucky/Bulk-Bing-Image-downloader "Bulk-Bing-Image-Downloader") to get raw image data from web.
2. Eliminate possible duplicate images by running detect_and_remove.py (adapted from [Adrian Rosebrock's code](https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/ "detect and remove duplicate images from a dataset")) 
3. Crop and align faces in the images by running align_faces.py (also adapted from [Adrian Rosebrock's face aligner](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/ "face alignment")

Note: Preprocessing must be performed on the dataset before training the model to maximize accuracy. To do this, 
