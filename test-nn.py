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



# Ex:

# LabelEncoder() should come fitted with an array of gestures already
# but this is needed as a test
gestures = ['goodbye', 'no', 'yes']
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(gestures)

# Gao should work on code to get an input video's feature data in the form of .npy w/ shape (20, 2048)
test = np.load("data/features/no37.npy")

predict_gesture(test, label_encoder)