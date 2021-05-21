#!/usr/bin/env python3
"""
Function that builds, trains, and saves a neural network
model in tensorflow using Adam optimization, mini-batch
gradient descent, learning rate decay, and batch normalization
"""
import tensorflow as tf
import numpy as np


def create_placeholders(nx, classes):
    """
    * nx: the number of feature columns in our data
    * classes: the number of classes in our classifier
    * Returns: placeholders named x and y, respectively
    * x is the placeholder for the input data to the neural network
    * y is the placeholder for the one-hot labels for the input data
    """
    x = tf.placeholder("float", shape=(None, nx), name='x')
    y = tf.placeholder("float", shape=(None, classes), name='y')
    return x, y

def create_layer(prev, n, activation):
    """
    Function that return the tensor output of the layer
    """
    initial = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=initial,
                            name='layer')
    return layer(prev)

def forward_prop(x, layer_sizes=[], activations=[]):
    """
    * x is the placeholder for the input data
    * layer_sizes is a list containing the number of nodes in
    each layer of the network
    * activations is a list containing the activation functions for
    each layer of the network
    * Returns: the prediction of the network in tensor form
    """
    pred = x
    for i in range(len(layer_sizes)):
        pred = create_layer(pred, layer_sizes[i], activations[i])
    return pred

def calculate_accuracy(y, y_pred):
    """
    * y is a placeholder for the labels of the input data
    * y_pred is a tensor containing the network’s predictions
    * Returns: a tensor containing the decimal accuracy of the prediction
    """
    y_max = tf.argmax(y, 1)
    y_pred_max = tf.argmax(y_pred, 1)
    evaluation = tf.equal(y_max, y_pred_max)
    accuracy = tf.reduce_mean(tf.cast(evaluation, "float"))
    return accuracy

def calculate_loss(y, y_pred):
    """
    * y is a placeholder for the labels of the input data
    * y_pred is a tensor containing the network’s predictions
    * Returns: a tensor containing the loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(y_pred, y)
    return loss

def shuffle_data(X, Y):
    """
    * X is the first numpy.ndarray of shape (m, nx) to shuffle
        * m is the number of data points
        * nx is the number of features in X
    * Y is the second numpy.ndarray of shape (m, ny) to shuffle
        * m is the same number of data points as in X
        * ny is the number of features in Y
    * Returns: the shuffled X and Y matrices
    """
    X_shuffled = X[np.random.permutation(len(X))]
    Y_shuffled = Y[np.random.permutation(len(Y))]
    return X_shuffled, Y_shuffled

def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    * loss is the loss of the network
    * alpha is the learning rate
    * beta1 is the weight used for the first moment
    * beta2 is the weight used for the second moment
    * epsilon is a small number to avoid division by zero
    * Returns: the Adam optimization operation
    """
    Adam_op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return Adam_op.minimize(loss)

def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    * alpha is the original learning rate
    * decay_rate is the weight used to determine the rate at which alpha will decay
    * global_step is the number of passes of gradient descent that have elapsed
    * decay_step is the number of passes of gradient descent that should occur before alpha is decayed further
    * the learning rate decay should occur in a stepwise fashion
    * Returns: the learning rate decay operation
    """
    learning_rate_de = tf.train.inverse_time_decay(alpha, global_step, decay_step, decay_rate, staircase=True)
    return learning_rate_de

def create_batch_norm_layer(prev, n, activation):
    """
    * prev is the activated output of the previous layer
    * n is the number of nodes in the layer to be created
    * activation is the activation function that should
      be used on the output of the layer
    * you should use the tf.layers.Dense layer as the base
      layer with kernal initializer
      tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    * your layer should incorporate two trainable parameters,
      gamma and beta, initialized as vectors of 1 and 0 respectively
    * you should use an epsilon of 1e-8
    * Returns: a tensor of the activated output for the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    out = tf.layers.Dense(units=n, kernel_initializer=init)
    x = out(prev)
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]), name="gamma")
    beta = tf.Variable(initial_value=tf.constant(0.0, shape=[n]), name="beta")
    mean, var = tf.nn.moments(x, axes=0)
    Z = tf.nn.batch_normalization(x, mean, var, beta, gamma, variance_epsilon=1e-8)
    if activation is None:
        return Z
    else:
        return activation(Z)

def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32,
          epochs=5, save_path='/tmp/model.ckpt'):
    """
    * Returns: the path where the model was saved
    * Your training function should allow for a smaller final batch (a.k.a. use the entire training set)
    * the learning rate should remain the same within the an epoch (a.k.a. all mini-batches within an epoch should use the same learning rate)
    """
    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]
    x, y = create_placeholders(nx, classes)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)
    lobal_step = tf.Variable(0)
    alpha_d = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_d, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
        with tf.Session() as sess:
        sess.run(init)
        m = X_train.shape[0]
        # mini batch definition
        if m % batch_size == 0:
            n_batches = int(m / batch_size)
        else:
            n_batches = int(m / batch_size + 1)
        for i in range(epochs + 1):
            cost_train = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            accuracy_train = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            cost_val = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_train))
            print("\tTraining Accuracy: {}".format(accuracy_train))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation Accuracy: {}".format(accuracy_val))
            if i < epochs:
                shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)
                for b in range(n_batches):
                    start = b * batch_size
                    end = (b + 1) * batch_size
                    if end > m:
                        end = m
                    X_mini_batch = shuffled_X[start:end]
                    Y_mini_batch = shuffled_Y[start:end]
                    next_train = {x: X_mini_batch, y: Y_mini_batch}
                    sess.run(train_op, feed_dict=next_train)
                    if (b + 1) % 100 == 0 and b != 0:
                        loss_mini_batch = sess.run(loss, feed_dict=next_train)
                        acc_mini_batch = sess.run(accuracy,
                                                  feed_dict=next_train)
                        print("\tStep {}:".format(b + 1))
                        print("\t\tCost: {}".format(loss_mini_batch))
                        print("\t\tAccuracy: {}".format(acc_mini_batch))
            sess.run(tf.assign(global_step, global_step + 1))
        return saver.save(sess, save_path)
