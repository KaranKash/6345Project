import tensorflow as tf
from utils import *
from nn import *
from model import *
from data import *
from load_data import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing

def evaluate_network(use_gpu=True, restore_if_possible=True, batch_size=30):
    g = tf.Graph()
    with g.as_default():
        with tf.device("/cpu:0"):
            # Build graph:
            image_batch, label_batch, num_examples_per_epoch = input_graph(training=False, batch_size=batch_size)
            mnist = tf.placeholder(tf.float32, shape=(batch_size*10, 4, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            nummatches = tf.placeholder(tf.int32, shape=(batch_size*10,))
            num_batches_per_epoch = num_examples_per_epoch // batch_size
            with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
                correct, loss, _, prediction= forward_propagation(image_batch, mnist, nummatches, train=False, dropout=False)
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
                    # evaluate
                    j = 0
                    tot_correct = 0
                    while j * batch_size*10 < num_examples_per_epoch and not coord.should_stop():
                        labels = sess.run([label_batch])
                        # print("labels",labels[0],labels)
                        labels = labels[0]
                        mnist_batch, nummatches_batch = generate_mnist_set(labels,train=False)
                        num_correct, preds = sess.run([correct, prediction], feed_dict={
                            mnist: mnist_batch, nummatches: nummatches_batch
                        })
                        tot_correct += num_correct
                        recordLosses(preds,nummatches_batch)
                        n0 = sum(x == 0 for x in nummatches_batch)
                        n1 = sum(x == 1 for x in nummatches_batch)
                        n2 = sum(x == 2 for x in nummatches_batch)
                        n3 = sum(x == 3 for x in nummatches_batch)
                        n4 = sum(x == 4 for x in nummatches_batch)
                        print("Batch %d/%d. Acc %.3f. Loss %.2f. Zeros %.2f. Ones %.2f. Twos %.2f. Threes %.2f. Fours %.2f." % (j+1, num_batches_per_epoch, 100*num_correct / float(batch_size*10), batch_loss, n0/float(batch_size*10), n1/float(batch_size*10), n2/float(batch_size*10), n3/float(batch_size*10), n4/float(batch_size*10)))
                        j += 1
                    total = j * batch_size
                    acc = tot_correct / float(total)

                except tf.errors.OutOfRangeError:
                    print('Done running!')
                finally:
                    # When done, ask the threads to stop.
                    coord.request_stop()

                # Wait for threads to finish.
                coord.join(threads)
                sess.close()
            return acc

def recordLosses(preds, actuals):
    with open('mistakes.txt','ab') as f:
        for i in range(len(preds)):
            p = preds[i]
            a = actuals[i]
            if p != a:
                f.write("Guessed: " + str(p) + ". Actually: " + str(a) + ".\n")

if __name__ == "__main__":
    acc = evaluate_network(use_gpu=True)
    print("Accuracy: " + str(acc))
