import cPickle as pickle
import numpy as np
import os

from skimage import feature
from skimage import color

ROOT = 'cifar-10-batches-py'

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data_for_knn():
    """ load all of cifar """
    print 'Loading data CIFAR10'
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))


    Xtr = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows becomes 50000 x 3072
    Xte = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xte_rows becomes 10000 x 3072


    return Xtr, Ytr, Xte, Yte


def load_data_for_nn():
    """ load all of cifar """
    print 'Loading data CIFAR10'
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))


    Xtr = np.array([np.reshape(x, (32*32*3, 1)) for x in Xtr])
    Xte = np.array([np.reshape(x, (32*32*3, 1)) for x in Xte])


    y_train = np.array([vectorized_result(y) for y in Ytr])
    y_test = np.array(Yte)

    return Xtr, y_train, Xte, y_test


def load_hog():
    print 'Loading HOGs'
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))

    y_train = np.array([vectorized_result(y) for y in Ytr])
    y_test = np.array(Yte)

    Xtr, Xte, y_train, y_test = Xtr[:3000], Xte[:1000], y_train[:3000], y_test[:1000]

    X_tr_hog, X_te_hog = [], []
    print 'Coverting train...'
    for x in Xtr:
        fd, hog_image = feature.hog(color.rgb2gray(x), orientations=8, pixels_per_cell=(2, 2),
                            cells_per_block=(1, 1), visualise=True)
        X_tr_hog.append(hog_image)
    print 'Converting test...'
    for x in Xte:
        fd, hog_image = feature.hog(color.rgb2gray(x), orientations=8, pixels_per_cell=(2, 2),
                            cells_per_block=(1, 1), visualise=True)
        X_te_hog.append(hog_image)

    X_tr_hog = np.array([np.reshape(x, (32*32, 1)) for x in X_te_hog])
    X_te_hog = np.array([np.reshape(x, (32*32, 1)) for x in X_te_hog])

    training_data = zip(X_tr_hog, y_train)
    test_data = zip(X_te_hog, y_test)

    return training_data, test_data
