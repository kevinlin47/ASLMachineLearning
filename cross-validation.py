import numpy as np
import os
import tensorflow
import cv2
import os
import argparse
import datetime
import h5py
import matplotlib.pyplot as plt


from time import time, gmtime, strftime
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from GestureRecognition.convert_all_videos import save_data_to_memory, get_data_from_memory

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils
from keras import optimizers

num_frames = 20
img_height = 640
img_width = 360
seed = 7
np.random.seed(seed)
gestures = ['goodbye', 'hey', 'no', 'yes']
num_classes = len(gestures)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
feature_length = 2048


if not os.path.exists("data/features"):
	os.makedirs("data/features")

if not os.path.exists("data/labels"):
	os.makedirs("data/labels")

# Run initially to make sure all the videos are saved into memory
# save_data_to_memory(gestures, 100)

# Get all our data from memory
x, y = get_data_from_memory(gestures, 10)

# Label encode the gesture strings
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(gestures)
y_label_encoded = label_encoder.transform(y)

# Transform the encoded y into one-hot arrays
# y = np_utils.to_categorical(y_label_encoded, num_classes)
y = y_label_encoded

input_shape = (num_frames, feature_length)
batch_size = 32
num_epochs = 32
count = 0
n_splits = 10

# Matlab Graph variables
accuracies = []



# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
cvscores = []

for train, test in kfold.split(x, y):
	# create model
	if (count < 1):
		y = np_utils.to_categorical(y, num_classes)
		count = 1

	model = Sequential()
	model.add(LSTM(2048, return_sequences=False,
				input_shape=input_shape,
				dropout=0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', 
		optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), 
		metrics=['accuracy'])

	# Fit the model
	model.fit(x[train], y[train], epochs=num_epochs, batch_size=batch_size, verbose=0)

	# evaluate the model
	scores = model.evaluate(x[train], y[train], verbose=0)

	accuracies.append(scores[1]*100)

	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



# Plot cross-validation results
fig, ax = plt.subplots()
start, end = ax.get_xlim()
x_range = np.arange(1,n_splits+1,1)
x_mean = np.arange(0,n_splits+1,1)

plt.xlabel('k-fold')
plt.ylabel('accuracy')
plt.title('k-fold cross-validation')

ax.xaxis.set_ticks(x_mean)
ax.set_xlim([0,n_splits+0.2])

x_mean = np.append(x_mean, n_splits+2)
y_mean = np.mean(cvscores)
y_mean = np.repeat(y_mean, len(x_mean))


acc_plot = ax.plot(x_range, accuracies, label='accuracy', marker='o', linestyle = 'None')
mean_line = ax.plot(x_mean, y_mean, label='mean', linestyle='--')
legend = ax.legend(loc='lower left')
plt.show()