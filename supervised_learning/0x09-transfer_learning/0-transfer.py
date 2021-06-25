#!/usr/bin/env python3
"""
Python script that trains a convolutional neural
network to classify the CIFAR 10 dataset
"""
import tensorflow.keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    * X is a numpy.ndarray of shape (m, 32, 32, 3) containing
    the CIFAR 10 data, where m is the number of data points
    * Y is a numpy.ndarray of shape (m,) containing the CIFAR
    10 labels for X
    * Returns: X_p, Y_p
        * X_p is a numpy.ndarray containing the preprocessed X
        * Y_p is a numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.xception.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == "__main__":
    (trainX, trainY), (testX, testY) = K.datasets.cifar10.load_data()
    trainX, trainY = preprocess_data(trainX, trainY)
    testX, testY = preprocess_data(testX, testY)

    input_t = K.Input(shape=(229, 229, 3))
    res_model = K.applications.Xception(include_top=True,
                                        weights="imagenet",
                                        input_tensor=input_t)
    for layer in res_model.layers[:-32]:
        layer.trainable = False
    for i, layer in enumerate(res_model.layers):
        print(i, layer.name, "-", layer.trainable)

    model = K.models.Sequential()
    model.add(K.layers.Lambda(lambda image: tf.image.resize(image, (229, 229))))
    model.add(res_model)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    checkpoint = K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                                             monitor='val_accuracy',
                                             mode='max',
                                             save_best_only=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])
    history = model.fit(trainX, trainY, batch_size=32, epochs=10, verbose=1,
                        validation_data=(testX, testY),
                        callbacks=[check_point])
    model.summary()
    model.save("cifar10.h5")
