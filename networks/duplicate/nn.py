import tensorflow as tf
from utils import *
import tflearn

def conv_layer(height, width, channels, name='conv1-layer', padding='SAME'):
    def make_layer(input_to_layer):
        with tf.variable_scope(name, values=[input_to_layer]) as scope:
            try:
                weights = weight_variable([height, width, input_to_layer.get_shape()[3], channels])
                bias = bias_variable([channels])
                variable_summaries(weights)
                variable_summaries(bias)
            except ValueError:
                scope.reuse_variables()
                weights = weight_variable([height, width, input_to_layer.get_shape()[3], channels])
                bias = bias_variable([channels])
        conv = tf.nn.conv2d(input_to_layer, weights, [1, 1, 1, 1], padding=padding)
        preactivation = tf.nn.bias_add(conv, bias)
        out = tf.nn.relu(preactivation)
        return out
    return make_layer

def pool_layer(height, width, vstride, hstride, name='pool1-layer', padding='SAME'):
    def make_layer(input_to_layer):
        return tf.nn.max_pool(input_to_layer, ksize=[1, height, width, 1], strides=[1, vstride, hstride, 1], padding=padding, name=name)
    return make_layer

def mean_pool_layer(name='pool1-layer', padding='SAME'):
    def make_layer(input_to_layer):
        return tflearn.layers.conv.global_avg_pool(input_to_layer, name=name)
    return make_layer

def norm_layer(name='norm1-layer'):
    def make_layer(input_to_layer):
        return tf.nn.local_response_normalization(input_to_layer, depth_radius=5, alpha=0.0001, beta=0.75, name=name)
    return make_layer

def flatten():
    def make_layer(inp):
        return tf.contrib.layers.flatten(inp)
    return make_layer

def fully_connected_layer(size, keep_prob=1.0, name='fc-layer'):
    def make_layer(input_to_layer):
        with tf.variable_scope(name, values=[input_to_layer]) as scope:
            try:
                weights = weight_variable([input_to_layer.get_shape()[1], size])
                bias = bias_variable([size])
                variable_summaries(weights)
                variable_summaries(bias)
            except ValueError:
                scope.reuse_variables()
                weights = weight_variable([input_to_layer.get_shape()[1], size])
                bias = bias_variable([size])
        preactivation = tf.matmul(input_to_layer, weights) + bias
        full_output = tf.nn.relu(preactivation)
        output = tf.nn.dropout(full_output, keep_prob=keep_prob)
        return output
    return make_layer

def softmax_layer(classes, name='softmax-layer'):
    def make_layer(input_to_layer):
        with tf.variable_scope(name, values=[input_to_layer]) as scope:
            fanin = input_to_layer.get_shape()[1]
            try:
                weights = weight_variable([fanin, classes])
                bias = bias_variable(classes)
                variable_summaries(weights)
                variable_summaries(bias)
            except ValueError:
                scope.reuse_variables()
                weights = weight_variable([fanin, classes])
                bias = bias_variable(classes)
            with tf.name_scope('preactivation'):
                preactivation = tf.matmul(input_to_layer, weights) + bias
                logits = preactivation
            with tf.name_scope('output'):
                with tf.name_scope('probabilities'):
                    proba = tf.nn.softmax(logits)
                with tf.name_scope('predictions'):
                    prediction = tf.argmax(proba, 1)
                    prediction = tf.to_int32(prediction, name='ToInt32')
        return input_to_layer, logits, proba, prediction
    return make_layer

def stack_layers(layers):
    def run_network(inp):
        state = inp
        for layer in layers:
            state = layer(state)
        return state
    return run_network
