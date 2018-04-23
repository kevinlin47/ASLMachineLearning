import numpy as np
import os
import tensorflow
import cv2
import os
import argparse
import datetime
import h5py
import pydot_ng
import graphviz
import matplotlib.pyplot as plt

from time import time, gmtime, strftime
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from GestureRecognition.convert_all_videos import save_data_to_memory, get_data_from_memory

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.utils import np_utils, plot_model
from keras import optimizers

num_frames = 20
img_height = 640
img_width = 360
seed = 7
np.random.seed(seed)
# gestures = ['goodbye', 'hey', 'no', 'yes']
gestures = ['find', 'goodbye', 'hey', 'no', 'yes']
num_classes = len(gestures)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
feature_length = 2048

# Create data folders for both features and labels to save into
if not os.path.exists("data/features"):
	os.makedirs("data/features")	

if not os.path.exists("data/labels"):
	os.makedirs("data/labels")


# Run initially to make sure all the videos are saved into memory
save_data_to_memory(gestures, 200)

# Get all our data from memory
x, y = get_data_from_memory(gestures, 200)

# Label encode the gesture strings
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(gestures)
y_label_encoded = label_encoder.transform(y)

# Transform the encoded y into one-hot arrays
y = np_utils.to_categorical(y_label_encoded, num_classes)

# Randomly shuffle x and y arrays to mix up the data, but keep row parity
s = np.arange(x.shape[0])
np.random.shuffle(s)
x = x[s]
y = y[s]

# Split x and y into train and test batches
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=seed)

input_shape = (num_frames, feature_length)
batch_size = 32
num_epochs = 16

# Initialize NN model structure
model = Sequential()
model.add(LSTM(2048, return_sequences=False,
               input_shape=input_shape,
               dropout=0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Output model structure as .png
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Compile model
model.compile(loss='categorical_crossentropy', 
	optimizer=optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), 
	metrics=['accuracy'])

# Print model structure
print(model.summary())

# Train model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate model
scores = model.evaluate(x, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Save model to refer to later
model.save('my_model.h5')

# Plot model accuracy results
acc = history.history['acc']
acc = [100*i for i in acc]
val_acc = history.history['val_acc']
val_acc = [100*i for i in val_acc]

fig, ax = plt.subplots()

x_range = np.arange(1,17,1)
ax.xaxis.set_ticks(x_range)
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.title('model accuracy')

# plt.plot(x_range, acc, marker='o')
ax.plot(x_range, val_acc, marker='o')
ax.legend(['accuracy'], loc='lower right')
plt.show()