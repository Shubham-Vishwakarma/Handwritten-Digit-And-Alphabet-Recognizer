import matplotlib.pyplot as plt
import numpy
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)
pt = 5000

(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
img = X_test[pt]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]


with open('cnn_mnist_model_for_mnist_digit.json','r') as json_file:
	loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("cnn_mnist_model_for_mnist_digit.h5")
print("Loaded saved model")

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

predi = model.predict_classes(X_test[pt].reshape(X_test[pt].shape[0], 1, 28, 28).astype('float32'))

plt.imshow(img, cmap=plt.get_cmap('gray'))
plt.title("Predicted: {},Truth: {}".format(predi[0],list(y_test[pt]).index(1)))
plt.show()