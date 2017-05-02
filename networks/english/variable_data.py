import os
import tensorflow as tf
import numpy as np
from load_data import *

DIR = os.path.dirname(os.path.realpath(__file__))

# For Audio
testdir = '../../English/test'
traindir = '../../English/train'

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 6796 # length 2-4
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 628 # length 2-4
IMAGE_WIDTH = 23
NUM_CHANNELS = 1

def variable_load_from_file(f):
    '''Given a file, returns a list of the string values in that value'''
    data = []
    for line in f:
        vector = []
        line = line.replace("[", "")
        line = line.replace("]", "")
        line_chars = line.split()
        for char in line_chars:
            vector.append(float(char))
        try:
            assert len(vector) == IMAGE_WIDTH
            data.append(vector)
        except AssertionError:
            if len(vector) == 0:
                pass
            else:
                # print len(vector)
                raise AssertionError

    #Convert data to a numpy array and invert it
    data = np.flipud(np.array(data,dtype=np.float32))
    return data.flatten().tolist()

def pad(spectrograms):
    reshaped = []
    for spec in spectrograms:
        spec = np.reshape(spec, (-1, 23, 1))
        spec = np.array(spec)
        reshaped.append(spec)
    maxlen = max([spec.shape[0] for spec in reshaped])
    out = []
    for i in range(len(reshaped)):
        spec = reshaped[i]
        cut = 1.*(maxlen - spec.shape[0])
        left = np.zeros((int(np.floor(cut/2)), IMAGE_WIDTH, NUM_CHANNELS))
        right = np.zeros((int(np.ceil(cut/2)), IMAGE_WIDTH, NUM_CHANNELS))
        spec = np.vstack((left,spec,right))
        out.append(spec)
    out = np.array(out)
    return out, maxlen

def variable_ld(rootdir,target):
    print("Couldn't find target file, creating it...")
    datastore = []
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            tmp = filename.split("_")
            chars = tmp[1][:-5]
            if len(chars) > 1 and len(chars) <= 4:
                f = open(os.path.join(subdir, filename))
                row = variable_load_from_file(f)
                f.close()
                data = [chars] + row
                datastore.append(data)
    print("Sorting the data by frame length...")
    datastore.sort(key=len)
    print("Finished sorting, writing to file...")
    with open(target, 'wb') as datafile:
        writer = csv.writer(datafile)
        for elem in datastore:
            writer.writerow(elem)
    print("Data preparation complete!")

def read_data_csv(target):
    print("Reading data from csv...")
    labels = []
    spectrograms = []
    with open(target, 'rb') as datafile:
        reader = csv.reader(datafile)
        for row in reader:
            labels.append(row[0])
            spectrograms.append(row[1:])
    print("Data reading complete!")
    return spectrograms, labels

# Graph ops for loading, parsing, and queuing training images
def variable_input_graph(partition='train'):
    with tf.name_scope("input"):
        if partition == 'train':
            usedir = traindir
            target = "vartrain.csv"
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

        elif partition == 'test':
            usedir = testdir
            target = "vartest.csv"
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        if not os.path.isfile(target):
            variable_ld(usedir,target)
            print("Created target file!")
        else:
            print("Located target file!")

        spectrograms, labels = read_data_csv(target)
        return spectrograms, labels, num_examples_per_epoch