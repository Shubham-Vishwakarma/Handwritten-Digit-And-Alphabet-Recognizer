from datasetReader import get_train_data_and_labels
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

data, labels = get_train_data_and_labels()

img1 = mpimg.imread(data[500])
img2 = mpimg.imread(data[1000])
img3 = mpimg.imread(data[2000])
img4 = mpimg.imread(data[3000])

# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(img1, cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(img2, cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(img3, cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(img4, cmap=plt.get_cmap('gray'))
# show the plot
plt.show()