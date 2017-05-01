import os
import tensorflow as tf
from utils import *
from nn import *
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_MODEL_DIR = os.path.join(DIR, "model")
SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.ckpt")
IMAGE_WIDTH = 23
NUM_CHANNELS = 1

def forward_propagation(images, mnist, nummatches, batch_size, maxlen, train=False, dropout=False):
    audio_network = stack_layers([
        # conv_layer(5, 23, 64, name='audio-conv1-layer',padding='VALID'),
        # norm_layer(name='audio-norm1-layer'),
        # pool_layer(3,1,1,1,name="audio-max-pool1-layer",padding='VALID'),
        # flatten(),
        # fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="audio-local1-layer"),
        # fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="audio-local2-layer")
        conv_layer(5, 23, 64, name='audio-conv1-layer',padding='VALID'),
        pool_layer(4,1,2,1,name="audio-max-pool1-layer",padding='VALID'),
        conv_layer(5, 1, 512, name='audio-conv2-layer',padding='VALID'),
        pool_layer(4,1,2,1,name="audio-max-pool2-layer",padding='VALID'),
        conv_layer(5, 1, 1024, name='audio-conv3-layer',padding='VALID'),
        mean_pool_layer(name="audio-mean-pool-layer",padding='VALID')
    ])

    image_network = stack_layers([
        conv_layer(5, 5, 32, name='image-conv1-layer'),
        pool_layer(2,2,2,2,name="image-max-pool1-layer"),
        conv_layer(5, 5, 64, name='image-conv2-layer'),
        pool_layer(2,2,2,2,name="image-max-pool2-layer"),
        flatten(),
        fully_connected_layer(512, keep_prob=0.5 if train and dropout else 1.0, name="image-local1-layer"),
        fully_connected_layer(512, keep_prob=1.0, name="image-local2-layer")
    ])

    image_joint_network = stack_layers([
        fully_connected_layer(1024, keep_prob=1.0, name="joint-local-layer"),
    ])

    classification_network = stack_layers([
        fully_connected_layer(512, keep_prob=1.0, name="class-local2-layer"),
        softmax_layer(5, name="class-softmax-layer")
    ])

    # length = images.get_shape()[1].value
    # print("LENGTH",length)
    # images = images.set_shape([batch_size, length, IMAGE_WIDTH, NUM_CHANNELS])
    specs = tf.concat([images]*10,0)
    mnistset = tf.unstack(mnist, axis=1)
    m1 = image_network(mnistset[0])
    m2 = image_network(mnistset[1])
    m3 = image_network(mnistset[2])
    m4 = image_network(mnistset[3])
    tmp = tf.concat([m1,m2,m3,m4],1)
    t1 = image_joint_network(tmp)
    # t1 = image_network(mnist)[0]
    print("SPECS",specs.shape)
    tmp = audio_network(specs)
    print("TMP",tmp.shape)
    t2 = tf.reshape(tmp, [batch_size*10, 1024])
    embeddings = tf.concat([t1,t2],1)
    # print("embeddings",embeddings.get_shape())
    _, logits, proba, prediction = classification_network(embeddings)

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            actual = nummatches
            with tf.name_scope('num_correct'):
                correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))

    with tf.name_scope('loss'):
        labels_one_hot = tf.one_hot(nummatches, 5, on_value=1.0, off_value=0.0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot))

    return correct, loss, proba, prediction
