import tensorflow as tf
from utils import *
from nn import *
from model import *
from variable_data import *
from load_data import *
import numpy as np
from multiprocessing import Pool
from contextlib import closing
import pickle

def optimizer():
    with tf.variable_scope("Optimizer"):
        global_step = tf.Variable(initial_value=0, trainable=False)
        increment_step = global_step.assign_add(1)
        opt = tf.train.AdamOptimizer(0.00001)
        return increment_step, opt, global_step

def train_network(training=True, use_gpu=True, restore_if_possible=True, batch_size=50):
    with tf.device("/cpu:0"):

        MAX_EPOCHS = 100.0
        eval_batch_size = 25
        eval_epochs = 5

        # setup metadata variables
        accuracies = []
        losses = []
        confusion = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[]}

        # Build graph training:
        all_spectrograms, all_labels, num_examples_per_epoch = variable_input_graph(training)
        # image_batch, label_batch, num_examples_per_epoch = input_graph(training=True, batch_size=batch_size)
        maximum = tf.placeholder(tf.int32, shape=())
        specs = tf.placeholder(tf.float32, shape=(batch_size, None, IMAGE_WIDTH, NUM_CHANNELS))
        mnist = tf.placeholder(tf.float32, shape=(batch_size*COPY, 9, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        nummatches = tf.placeholder(tf.int32, shape=(batch_size*COPY,))
        num_batches_per_epoch = num_examples_per_epoch // batch_size
        increment_step, opt, step = optimizer()
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            correct, loss, _, __ = forward_propagation(specs, mnist, nummatches, batch_size, maximum, train=True, dropout=True)
            grads = opt.compute_gradients(loss)
        with tf.control_dependencies([opt.apply_gradients(grads), increment_step]):
            train = tf.no_op(name='train')

        # Build graph eval:
        eval_spectrograms, eval_labels, e_num_examples_per_epoch = variable_input_graph(False)
        e_maximum = tf.placeholder(tf.int32, shape=())
        e_specs = tf.placeholder(tf.float32, shape=(eval_batch_size, None, IMAGE_WIDTH, NUM_CHANNELS))
        e_mnist = tf.placeholder(tf.float32, shape=(eval_batch_size*COPY, 9, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        e_nummatches = tf.placeholder(tf.int32, shape=(eval_batch_size*COPY,))
        e_num_batches_per_epoch = e_num_examples_per_epoch // eval_batch_size
        e_increment_step, _, e_step = optimizer()
        with tf.device("/gpu:0" if use_gpu else "/cpu:0"):
            e_correct, _, __, e_predictions = forward_propagation(e_specs, e_mnist, e_nummatches, eval_batch_size, e_maximum, train=False, dropout=False)
        with tf.control_dependencies([e_increment_step]):
            test = tf.no_op(name='test')

        # summaries = tf.summary.merge_all()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)) as sess:
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
                for b in range(int(num_examples_per_epoch // batch_size * MAX_EPOCHS)):
                    # offset = (b * batch_size) % (num_examples_per_epoch - batch_size)
                    offset = np.random.randint(0,num_examples_per_epoch - batch_size)
                    spectrograms = all_spectrograms[offset:(offset + batch_size)]
                    spectrograms, maxlen = pad(spectrograms)
                    labels = all_labels[offset:(offset + batch_size)]
                    mnist_batch, nummatches_batch = generate_mnist_set(labels, COPY)
                    _, num_correct, batch_loss, i = sess.run([train, correct, loss, step], feed_dict={
                        specs: spectrograms, mnist: mnist_batch, nummatches: nummatches_batch, maximum: maxlen
                    })
                    in_batch = i % num_batches_per_epoch
                    if in_batch == 0:
                        in_batch = num_batches_per_epoch
                    acc = 100*num_correct / float(batch_size*COPY)
                    n0 = sum(x == 0 for x in nummatches_batch)
                    n1 = sum(x == 1 for x in nummatches_batch)
                    n2 = sum(x == 2 for x in nummatches_batch)
                    n3 = sum(x == 3 for x in nummatches_batch)
                    n4 = sum(x == 4 for x in nummatches_batch)
                    n5 = sum(x == 5 for x in nummatches_batch)
                    n6 = sum(x == 6 for x in nummatches_batch)
                    n7 = sum(x == 7 for x in nummatches_batch)
                    print("Train. Epoch %d. Batch %d/%d. Acc %.3f. Loss %.2f. Zeros %.2f. Ones %.2f. Twos %.2f. Threes %.2f. Fours %.2f. Fives %.2f. Sixes %.2f. Sevens %.2f." % (epoch_count, in_batch, num_batches_per_epoch, acc, batch_loss, n0/float(batch_size*COPY), n1/float(batch_size*COPY), n2/float(batch_size*COPY), n3/float(batch_size*COPY), n4/float(batch_size*COPY), n5/float(batch_size*COPY), n6/float(batch_size*COPY), n7/float(batch_size*COPY)))
                    epoch_count = (i // (num_batches_per_epoch)) + 1

                    if in_batch == num_batches_per_epoch:
                        accuracies.append(acc)
                        losses.append(batch_loss)

                print("Done training. Saving metadata & evaluating...")
                with open('./results/accuracies.pkl', 'wb') as f:
                    pickle.dump(accuracies, f, protocol=2)
                with open('./results/losses.pkl', 'wb') as f:
                    pickle.dump(losses, f, protocol=2)

                # evaluation
                for b in range(int(e_num_examples_per_epoch // eval_batch_size * eval_epochs)):
                    offset = np.random.randint(0,e_num_examples_per_epoch - eval_batch_size)
                    spectrograms = eval_spectrograms[offset:(offset + eval_batch_size)]
                    spectrograms, maxlen = pad(spectrograms)
                    labels = eval_labels[offset:(offset + eval_batch_size)]
                    mnist_batch, nummatches_batch = generate_mnist_set(labels, COPY, train=False)
                    _, num_correct, i, preds = sess.run([test, e_correct, e_step, e_predictions], feed_dict={
                        e_specs: spectrograms, e_mnist: mnist_batch, e_nummatches: nummatches_batch, e_maximum: maxlen
                    })
                    confusion = record(nummatches_batch, preds, confusion)
                    # in_batch = i % e_num_batches_per_epoch
                    # if in_batch == 0:
                    #     in_batch = e_num_batches_per_epoch
                    n0 = sum(x == 0 for x in nummatches_batch)
                    n1 = sum(x == 1 for x in nummatches_batch)
                    n2 = sum(x == 2 for x in nummatches_batch)
                    n3 = sum(x == 3 for x in nummatches_batch)
                    n4 = sum(x == 4 for x in nummatches_batch)
                    n5 = sum(x == 5 for x in nummatches_batch)
                    n6 = sum(x == 6 for x in nummatches_batch)
                    n7 = sum(x == 7 for x in nummatches_batch)
                    print("Test Acc %.3f. Zeros %.2f. Ones %.2f. Twos %.2f. Threes %.2f. Fours %.2f. Fives %.2f. Sixes %.2f. Sevens %.2f." % (100*num_correct / float(eval_batch_size*COPY), n0/float(eval_batch_size*COPY), n1/float(eval_batch_size*COPY), n2/float(eval_batch_size*COPY), n3/float(eval_batch_size*COPY), n4/float(eval_batch_size*COPY), n5/float(eval_batch_size*COPY), n6/float(eval_batch_size*COPY), n7/float(eval_batch_size*COPY)))

                with open('./results/confusion.pkl', 'wb') as f:
                    pickle.dump(confusion, f, protocol=2)

            except tf.errors.OutOfRangeError:
                print('Done running!')
            finally:
                # When done, ask the threads to stop.
                coord.request_stop()

            # Wait for threads to finish.
            coord.join(threads)
            sess.close()

def record(actuals, preds, dic):
    for i in range(len(actuals)):
        a = actuals[i]
        p = preds[i]
        dic[a].append(p)
    return dic

if __name__ == "__main__":
    train_network(use_gpu=False)
