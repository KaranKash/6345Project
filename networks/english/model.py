import os
import tensorflow as tf
from utils import *
from nn import *
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_MODEL_DIR = os.path.join(DIR, "model")
SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.ckpt")

def forward_propagation(images, mnist, nummatches, train=False, dropout=False):
    audio_network = stack_layers([
        conv_layer(5, 23, 64, name='audio-conv1-layer'),
        pool_layer(3,4,1,2,name="audio-max-pool1-layer"),
        flatten(),
        fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="audio-local1-layer"),
        fully_connected_layer(512, keep_prob=1.0, name="audio-local2-layer"),
        softmax_layer(11, name="audio-softmax-layer")
    ])

    image_network = stack_layers([
        conv_layer(5, 5, 32, name='image-conv1-layer'),
        pool_layer(2,2,2,2,name="image-max-pool1-layer"),
        conv_layer(5, 5, 64, name='image-conv2-layer'),
        pool_layer(2,2,2,2,name="image-max-pool2-layer"),
        flatten(),
        fully_connected_layer(512, keep_prob=0.5 if train and dropout else 1.0, name="image-local1-layer"),
        fully_connected_layer(512, keep_prob=1.0, name="image-local2-layer"),
        softmax_layer(10, name="image-softmax-layer")
    ])

    classification_network = stack_layers([
        fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="class-local1-layer"),
        fully_connected_layer(512, keep_prob=1.0, name="class-local2-layer"),
        softmax_layer(5, name="class-softmax-layer")
    ])
    specs = tf.concat(concat_dim=0,values=[images]*10)
    print("specs",specs.get_shape())
    t1 = image_network(mnist)[0]
    t2 = audio_network(specs)[0]
    embeddings = tf.concat(concat_dim=1, values=[t1,t2])
    print("embeddings",embeddings.get_shape())
    _, logits, proba, prediction = classification_network(embeddings)

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            actual = nummatches
            with tf.name_scope('num_correct'):
                correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))

    with tf.name_scope('loss'):
        labels_one_hot = tf.one_hot(nummatches, 5, on_value=1.0, off_value=0.0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot))

    return correct, loss, proba
