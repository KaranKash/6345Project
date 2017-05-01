import tensorflow as tf
from utils import *
from nn import *
from model import *
from data import *
from variable_data import *
from load_data import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing

MAX_EPOCHS = 1.0

def optimizer():
    with tf.variable_scope("Optimizer"):
        global_step = tf.Variable(initial_value=0, trainable=False)
        increment_step = global_step.assign_add(1)
        opt = tf.train.AdamOptimizer(0.00001)
        return increment_step, opt, global_step

def train_network(partition="train", use_gpu=True, restore_if_possible=True, batch_size=50):
    with tf.device("/cpu:0"):
        # Build graph:
        all_spectrograms, all_labels, num_examples_per_epoch = variable_input_graph(partition)
        # image_batch, label_batch, num_examples_per_epoch = input_graph(training=True, batch_size=batch_size)
        maximum = tf.placeholder(tf.int32, shape=())
        specs = tf.placeholder(tf.float32, shape=(batch_size, None, IMAGE_WIDTH, NUM_CHANNELS))
        mnist = tf.placeholder(tf.float32, shape=(batch_size*10, 4, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        nummatches = tf.placeholder(tf.int32, shape=(batch_size*10,))
        num_batches_per_epoch = num_examples_per_epoch // batch_size
        increment_step, opt, step = optimizer()
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            correct, loss, _, __ = forward_propagation(specs, mnist, nummatches, batch_size, maximum, train=True, dropout=True)
            grads = opt.compute_gradients(loss)
        with tf.control_dependencies([opt.apply_gradients(grads), increment_step]):
            train = tf.no_op(name='train')
        summaries = tf.summary.merge_all()

        # Train:
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            summary_writer = tf.summary.FileWriter('tflog', sess.graph)  # For logging for TensorBoard

            saver = tf.train.Saver()
            with tf.device("/cpu:0"):
                sess.run(tf.global_variables_initializer())

            if restore_if_possible:
                try:
                    saver.restore(sess, tf.train.latest_checkpoint(SAVED_MODEL_DIR))
                    print("Found in-progress model. Will resume from there.")
                except:
                    print("Couldn't find old model. Starting from scratch.")

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            epoch_count = 1
            try:
                # training
                for b in range(int(100*NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) // batch_size):
                    offset = (b * batch_size) % (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN - batch_size)
                    spectrograms = all_spectrograms[offset:(offset + batch_size)]
                    spectrograms, maxlen = pad(spectrograms)
                    labels = all_labels[offset:(offset + batch_size)]
                    mnist_batch, nummatches_batch = generate_mnist_set(labels)
                    _, num_correct, batch_loss, i = sess.run([train, correct, loss, step], feed_dict={
                        specs: spectrograms, mnist: mnist_batch, nummatches: nummatches_batch, maximum: maxlen
                    })
                    in_batch = i % num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = num_batches_per_epoch
                    epoch_count = ((i-1) // (num_batches_per_epoch)) + 1
                    n0 = sum(x == 0 for x in nummatches_batch)
                    n1 = sum(x == 1 for x in nummatches_batch)
                    n2 = sum(x == 2 for x in nummatches_batch)
                    n3 = sum(x == 3 for x in nummatches_batch)
                    n4 = sum(x == 4 for x in nummatches_batch)
                    print("Epoch %d. Batch %d/%d. Acc %.3f. Loss %.2f. Zeros %.2f. Ones %.2f. Twos %.2f. Threes %.2f. Fours %.2f." % (epoch_count, in_batch, num_batches_per_epoch, 100*num_correct / float(batch_size*10), batch_loss, n0/float(batch_size*10), n1/float(batch_size*10), n2/float(batch_size*10), n3/float(batch_size*10), n4/float(batch_size*10)))

                    if in_batch == num_batches_per_epoch:
                        # Checkpoint, save the model:
                        summary = sess.run(summaries)
                        summary_writer.add_summary(summary)
                        print("Saving to %s" % SAVED_MODEL_PATH)
                        saver.save(sess, SAVED_MODEL_PATH, global_step=i)
                        # evaluate(partition="test")

            except tf.errors.OutOfRangeError:
                print('Done running!')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

if __name__ == "__main__":
    train_network(use_gpu=False)
