import warnings
import numpy as np
import os
import tensorflow
import cv2
import os
import argparse
import datetime
import h5py

from sklearn import preprocessing

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from extractor import Extractor


"""
Load Neural Network model and feed it an input np array to predict
an output gesture

:param npy array video_data: features data as an .npy array with size (20, 2048) 
:parm LabelEncoder() label_encoder: pre-intialized LabelEncoder class with parameters we need

:return: None 
:rtype: void

"""

def predict_gesture(video_data , label_encoder):

	label_encoder = label_encoder

	model = load_model('my_model.h5')

	input_data = video_data 

	pad = np.random.rand(20,2048)

	input_data = np.append(input_data, pad)
	input_data = np.reshape(input_data, (2, 20, 2048))

	prediction = model.predict_classes(input_data[0:1])
	prediction = label_encoder.inverse_transform(prediction)

	print('The Model predicts the gesture is: ' + prediction[0].upper())

if not os.path.exists("test_data"):
    os.makedirs("test_data")

model = Extractor()
parser = argparse.ArgumentParser() 
filename = None 
parser.add_argument('-f', '--file', dest="file", default="") 
options = parser.parse_args() 
if options.file: 
   filename = options.file 

if filename is None:
   print('Please enter a directory of frames.')
   exit()

sequence = []
for frame in range(1,21):
   file = filename + "/frame" + str(frame) + ".jpg"
   print(file)
   try:
      features = model.extract(file)
   except IOError:
      print("Could not extract file " + file)
   sequence.append(features)
   np.save("test_data/"+filename + ".npy", sequence)

# Ex:

# LabelEncoder() should come fitted with an array of gestures already
# but this is needed as a test
gestures = ['goodbye', 'no', 'yes']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(gestures)

# Gao should work on code to get an input video's feature data in the form of .npy w/ shape (20, 2048)
test = np.load("test_data/" + filename + ".npy")

predict_gesture(test, label_encoder)
