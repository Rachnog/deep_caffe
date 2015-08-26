import mnist_loader
import cifar_loader
import numpy as np
from random import randint
from pylab import *


class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            # distances = np.sum(np.abs(self.Xtr - X[i, :]))

            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)

            min_index = np.argmin(distances)  # get the index with smallest distance
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example

        return Ypred


# MNSIT load
X_train, y_train, X_test, y_test = mnist_loader.load_data_for_knn()


# CIFAR load
#X_train, y_train, X_test, y_test = cifar_loader.load_data_for_knn()

nn = KNearestNeighbor()

print 'Fitting classifier'
nn.train(X_train, y_train)

res = 0.0
for i in range(100):
    index = randint(0, len(X_test))
    im = np.array([X_test[index]])
    print nn.predict(im), y_test[index]
    if nn.predict(im)[0] == y_test[index]:
        res += 1

print res / 100
