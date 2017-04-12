from __future__ import print_function

import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import random
import random2

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
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
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

def generateRandomOutput(N):
    print("Generating random input of size " + str(N))
    theString = ''
    for i in range(N):
        theString += str(int(10 * random.random()))
    return generateOutputGivenInput(theString)

def generateOutputGivenInput(chars):
    print("Creating sequence")
    N = len(chars)
    output = np.zeros((N, 28, 28), dtype = np.float)

    for i in range(N):
        loc = np.where(y_train == int(chars[i]))
        L = len(loc[0])
        index = random2.randint(0, L - 1)
        temp = X_train[ loc[0][index], :, :, : ]
        output[i, :, :] = temp

    return output

def draw(output, random = False):
    N = len(output)

    fig = plt.figure(1, (6., 6.))
    grid = ImageGrid(fig, 110, nrows_ncols = (1, N), axes_pad = 0.0)

    gridIndexes = [i for i in range(N)]

    if random:
        print("Randomizing sequence")
        random2.shuffle(gridIndexes)
    for i in gridIndexes:
        grid[i].imshow( output[gridIndexes[i]], cmap = "gray" )

    print("Drawing output")
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        argument = sys.argv[1]
        if len(sys.argv[1]) == 1:
            draw(generateRandomOutput(int(argument)))
        else:
            draw(generateOutputGivenInput(sys.argv[1]))
    elif len(sys.argv) == 3:
        draw(generateOutputGivenInput(sys.argv[1]), sys.argv[2])
    else:
        draw(generateRandomOutput(4))

