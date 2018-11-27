from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[200], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[360], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[600], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[750], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()