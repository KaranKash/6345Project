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
IMAGE_HEIGHT = 400
NUM_CHANNELS = 1
NUM_CLASSES = 11
IMAGE_SIZE = 28

def read_from_csv(filename_queue):
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    record_defaults = [["0"]] + [[0.0]]*9200
    outlist = tf.decode_csv(csv_row, record_defaults=record_defaults)
    image = tf.stack(outlist[1:])
    label = outlist[0]
    return image, label

# Graph ops for loading, parsing, and queuing training images
def input_graph(training=True, partition='test', batch_size=256):
    with tf.name_scope("input"):
        if training or partition == 'train':
            usedir = traindir
            target = "train2.csv"
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

        elif partition == 'test':
            usedir = testdir
            target = "test2.csv"
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        if not os.path.isfile(target):
            ld(usedir,target)
            print("Created target file!")
        else:
            print("Located target file!")

        # Organizing audio data into spectrogram images and labels
        filename_queue = tf.train.string_input_producer([target])
        image, record_label = read_from_csv(filename_queue)
        image = tf.cast(tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS]),tf.float32)

        with tf.name_scope("batching"):
            # Load set of images to start, then continue enqueuing up to capacity
            min_after_dequeue = batch_size*5
            capacity = min_after_dequeue + 3 * batch_size
            kwargs = dict(batch_size=batch_size, capacity=capacity)
            if training:
                batch_fn = tf.train.shuffle_batch
                kwargs["min_after_dequeue"] = min_after_dequeue
            else:
                batch_fn = tf.train.batch
            image_batch, label_batch = batch_fn([image, record_label], **kwargs)

            # The examples and labels for training a single batch
            tf.summary.image("image", image_batch, max_outputs=3)
            return image_batch, label_batch, num_examples_per_epoch
