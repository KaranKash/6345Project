import tensorflow as tf
from utils import *
from nn import *
from model import *
from data import *
from load_data import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing

MAX_EPOCHS = 1.0

def optimizer(num_batches_per_epoch):
    with tf.variable_scope("Optimizer"):
        global_step = tf.Variable(initial_value=0, trainable=False)
        increment_step = global_step.assign_add(1)
        opt = tf.train.AdamOptimizer(0.00001)
        return increment_step, opt, global_step

def train_network(use_gpu=True, restore_if_possible=True, batch_size=128):
    with tf.device("/cpu:0"):
        # Build graph:
        image_batch, label_batch, num_examples_per_epoch = input_graph(training=True, batch_size=batch_size)
        mnist = tf.placeholder(tf.float32, shape=(batch_size*10, IMAGE_SIZE*2, IMAGE_SIZE*2, NUM_CHANNELS))
        nummatches = tf.placeholder(tf.int32, shape=(batch_size*10,))
        num_batches_per_epoch = num_examples_per_epoch // batch_size
        increment_step, opt, step = optimizer(num_batches_per_epoch)
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            correct, loss, _ = forward_propagation(image_batch, mnist, nummatches, train=True, dropout=True)
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
                while not coord.should_stop():
                    labels = sess.run([label_batch])
                    # print("labels",labels[0],labels)
                    labels = labels[0]
                    mnist_batch, nummatches_batch = generate_mnist_set(labels)
                    _, num_correct, batch_loss, i = sess.run([train, correct, loss, step], feed_dict={
                        mnist: mnist_batch, nummatches: nummatches_batch
                    })
                    in_batch = i % num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = num_batches_per_epoch
                    epoch_count = ((i-1) // (num_batches_per_epoch)) + 1

                    print("Epoch %d. Batch %d/%d. Acc %.3f. Loss %.2f" % (epoch_count, in_batch, num_batches_per_epoch, 100*num_correct / float(batch_size*10), batch_loss))

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
    train_network(use_gpu=True)
