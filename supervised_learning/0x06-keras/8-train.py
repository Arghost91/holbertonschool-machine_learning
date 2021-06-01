#!/usr/bin/env python3
"""
Update the function to also save the best iteration of the model
"""
import tensorflow.keras as k


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    * save_best is a boolean indicating whether to save the model after
    each epoch if it is the best
        * a model is considered the best if its validation loss is the
        lowest that the model has obtained
    * filepath is the file path where the model should be saved
    """
    callbacks = []

    if validation_data and early_stopping:
        callbacks.append(k.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience))

    if validation_data and learning_rate_decay:
        def slearning_rate_decay(epoch):
            return alpha / (1 + (decay_rate * epoch))

        callbacks.append(k.callbacks.LearningRateScheduler(
            schedule=schedule, verbose=1))

    if save_best:
        callbacks.append(k.callbacks.ModelCheckpoint(
            filepath=filepath, monitor='val_loss', verbose=0,
            save_best_only=True, save_weights_only=False,
            mode='auto', period=1))

    train = network.fit(data, labels, batch_size=batch_size, epochs=epochs, 
                        verbose=verbose, shuffle=shuffle,
                        validation_data=validation_data, callbacks=callbacks)

    return train
