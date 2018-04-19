import numpy as np
import cv2
import os, os.path
import random
import argparse
import keras.preprocessing.image as kpi
from PIL import Image
from extractor import Extractor

"""
Go through all video folders and run frames through InceptionV3 Neural Network
and save the data in 'data/features' and 'data/labels' folders as .npy 

:param str[] gestures: gesture titles
:param int num_videos: total number of videos for each gesture
:parm int num_frames: desired number of frames per video

:return: None
:rtype: void

"""

def save_data_to_memory(gestures, num_videos = 5, num_frames = 20):
    model = Extractor()
    classifiers = np.array(gestures)
    array_of_classifiers = np.array([])
    array_of_features = np.array([])

    for classifier in classifiers:
        for num in range(1,num_videos+1):
            sequence = []
            file = "video/"+classifier+str(num)
            total_frames = len([name for name in os.listdir(file) if os.path.isfile(os.path.join(file,name))])
            ran_range = random.sample(range(1,total_frames),num_frames)
            ran_range.sort()
            for frame in ran_range:

                file = file+"/frame"+str(frame)+".jpg"
                print(file)

                if os.path.isfile("data/features/"+classifier+str(num) + ".npy"):
                    continue
                    print("skipped cause data's been saved already")

                # Feed the .jpg file into model and extract feature data
                try:
                    features = model.extract(file)
                except IOError:
                    print("Could not extract file "+classifier+str(num))

                # Append the 2048 features 'num_frames' amount of times
                sequence.append(features)

            # Check if data has been saved
            if os.path.isfile("data/features/"+classifier+str(num) + ".npy"):
                continue

            # Save features and labels data
            else:
                np.save("data/features/"+classifier+str(num), sequence)
                np.save("data/labels/"+classifier+str(num), classifier)

"""
Load .npy frame data that's saved in 'data/features' and 'data/labels' folders

:param str[] gestures: gesture titles
:param int num_videos: total number of videos for each gesture
:parm int num_frames: desired number of frames per video

:return: (np array, np array)
:rtype: tuple

"""

def get_data_from_memory(gestures = ['goodbye', 'no', 'yes'], num_videos = 5, num_frames = 20):
    classifiers = np.array(gestures)
    number_of_videos = num_videos
    number_of_frames = num_frames

    array_of_features = np.array([])
    array_of_classifiers = np.array([])

    for classifier in classifiers:
        for num in range(1,number_of_videos+1):
            x = np.load("data/features/"+str(classifier)+str(num)+".npy")
            x = x.flatten()
            y = np.load("data/labels/"+str(classifier)+str(num)+".npy")
            y = y.flatten()
            array_of_features = np.append(array_of_features, x)
            array_of_classifiers = np.append(array_of_classifiers, y)

    array_of_features =  np.reshape(array_of_features, (number_of_videos*len(gestures) , number_of_frames, 2048))
    array_of_classifiers = np.reshape(array_of_classifiers, (number_of_videos*len(gestures)))

    return array_of_features, array_of_classifiers
