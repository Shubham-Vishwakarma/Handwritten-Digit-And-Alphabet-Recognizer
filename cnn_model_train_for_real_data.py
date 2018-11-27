import numpy
import math
from scipy import ndimage
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
from datasetReader import get_train_data_and_labels
from myutils import pre_process_image
import cv2 as cv
from random import shuffle
K.set_image_dim_ordering('th')

data, labels = get_train_data_and_labels()
print("Reading Data completed")

c = list(zip(data,labels))
shuffle(c)

data,labels = zip(*c)

X = []
Y = []

print("Beginning with pre-processing of image")

for filename,label in zip(data, labels):
	img_gray = pre_process_image(filename=filename)
	X.append(img_gray)
	Y.append(label)

print("Completed with pre-processing of image")

X = numpy.array(X)
Y = numpy.array(Y)

X_train = X
y_train = Y
X_test = X
y_test = Y

# X_train = X[0: 1144, :]
# y_train = Y[0: 1144]
# X_test = X[1145: 1430, :]
# y_test = Y[1145: 1430]
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]


def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
	
	
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


model_json = model.to_json()
with open("cnn_model_for_real_data.json","w") as json_file:
	json_file.write(model_json)

model.save_weights("cnn_model_for_real_data.h5")
print("Saved model")