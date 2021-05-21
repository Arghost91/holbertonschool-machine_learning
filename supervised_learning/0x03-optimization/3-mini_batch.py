#!/usr/bin/env python3
"""
Function that trains a loaded neural network model
using mini-batch gradient descent
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    * Trains a loaded neural network model using
      mini-batch gradient descent
    * Returns: the path where the model was saved
    """
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        train_op = tf.get_collection('train_op')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        m = X_train.shape[0]
        if m % batch_size == 0:
            batches = int(m / batch_size)
        else:
            batches = int(m / batch_size) + 1

        for i in range(epochs + 1):
            training_cost, training_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})
            validation_cost, validation_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(training_cost))
            print("\tTraining Accuracy: {}".format(training_accuracy))
            print("\tValidation Cost: {}".format(validation_cost))
            print("\tValidation Accuracy: {}".format(validation_accuracy))

            if i < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
 
                for j in range(batches):
                    batch_start = j * batch_size
                    batch_end = batch_start + batch_size
                    if batch_end > m:
                        batch_end = m
                    batch_X = X_shuffled[batch_start:batch_end]
                    batch_Y = Y_shuffled[batch_start:batch_end]
                    sess.run(train_op, feed_dict={x: batch_X, y: batch_Y})

                    if (j + 1) % 100 == 0 and j != 0:
                        batch_cost, batch_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: batch_X, y: batch_Y})
                        print("\tStep {}".format(j + 1))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy: {}".format(batch_accuracy))
        return saver.save(sess, save_path)
