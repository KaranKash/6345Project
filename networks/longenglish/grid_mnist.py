from __future__ import print_function

import sys
import os
import numpy as np
import random

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('./data/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('./data/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('./data/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('./data/t10k-labels-idx1-ubyte.gz')

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_test, y_test


print("Loading MNIST data...")
X_train, y_train, X_test, y_test = load_dataset()

def makeGrid(chars, train=True, N=3):
    ## make sure that len(chars) <= N*N
    if len(chars) <= int(N) * int(N):
        output = np.zeros((int(N) * int(N), 28, 28), dtype= np.float)

        if train:
            y = y_train
            X = X_train
        else:
            y = y_test
            X = X_test

        for i in range(len(chars)):
            loc = np.where(y == int(chars[i]))
            L = len(loc[0])
            index = random.randint(0, L - 1)
            temp = X[ loc[0][index], :, :, : ]
            output[i, :, :] = temp

        # fig = plt.figure(1, (6., 6.))
        # grid = ImageGrid(fig, 110, nrows_ncols=(int(N), int(N)), axes_pad=0.0)

        gridIndexes = []
        for i in range (int(N) * int(N)):
            gridIndexes.append(i)

        random.shuffle(gridIndexes)

        # for i in gridIndexes:
        #     grid[i].imshow( output[gridIndexes[i]], cmap='gray' )

        out = [output[gridIndexes[i]] for i in range(len(gridIndexes))]

        return out

        # row1 = np.hstack((output[gridIndexes[0]],output[gridIndexes[1]]))
        # row2 = np.hstack((output[gridIndexes[2]],output[gridIndexes[3]]))
        # out = np.vstack((row1,row2))
        # return out

    else:
        print("Please ensure proper input dimensions")
        return None
