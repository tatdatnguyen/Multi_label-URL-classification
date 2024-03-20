from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, Activation, Conv1D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import numpy as np
import random
import os
from config.config import SEED_VALUE, LEARNING_RATE

import logging

logging.basicConfig(filename='training_log.log', level=logging.DEBUG)


# Get the model
def get_model(n_outputs, vocab_size, max_len):
    """
    To get the model.

    :param n_outputs: output shape
    :param vocab_size: vocabulary size
    :param max_len: maximum sentence length
    :return: model ready to train
    """
    filter_length = 300

    model = Sequential()
    model.add(Embedding(vocab_size, 20))
    model.add(Dropout(0.1))
    model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())

    model.add(Dense(n_outputs))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adam(LEARNING_RATE), loss='binary_crossentropy', metrics=['acc'])
    model.summary()

    logging.info('Model created !')

    return model


# Evaluate the model
def train_model(x,
                y,
                epochs, batch_size, validation_split,
                vocab_size, max_len,
                trained_model_output_path, model_name):
    """
    To train the model.

    :param x: train data
    :param y: train label data
    :param epochs: epoch number
    :param batch_size: batch size
    :param validation_split: validation split number
    :param vocab_size: vocabulary size in all the corpus
    :param max_len: maximum word length
    :param trained_model_output_path: output path to save the model
    :param model_name: model name
    :return: trained model
    """
    # Set the seed
    np.random.seed(SEED_VALUE)
    random.seed(SEED_VALUE)
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)

    n_inputs, n_outputs = x.shape[1], y.shape[1]

    # Get the model
    model = get_model(n_outputs, vocab_size, max_len)

    # Fit model
    callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=3),
        ModelCheckpoint(filepath=f'{trained_model_output_path}/{model_name}', save_best_only=True)
    ]

    history = model.fit(x,
                        y,
                        verbose=1,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=validation_split,
                        callbacks=callbacks)

    logging.info('The model is successfully saved.')

    return history


def evaluate_model(history, x_test, y_test):
    """
    To evaluate the model.

    :param history: model history
    :param x_test: test data
    :param y_test: test label
    :return: None
    """
    results = history.model.evaluate(x_test, y_test, batch_size=64)
    print("Testing on data...\n")
    print(f"Test loss : {results[0]}")
    print(f"Test acc : {results[1]}")

    logging.info(f'Evaluating the model is completed. Test loss is {results[0]} and test acc is {results[1]}.')

    return results


def get_jaccard_score(history, x_test, y_test):
    """
    To get the jaccard score.

    :param history: model history
    :param x_test: data to predict
    :param y_test: true labels
    :return: Jaccard score
    """
    # predict the labels
    y_pred = history.model.predict(x_test)
    # Get an array with 1 or 0 for each classes predicted
    y_pred_1 = y_pred >= 0.5

    intersection = np.logical_and(y_test, y_pred_1)
    union = np.logical_or(y_test, y_pred_1)
    jaccard_score = np.sum(intersection) / np.sum(union)

    print(f"Jaccard score : {jaccard_score}")

    logging.info(f'Jaccard score is {jaccard_score}')

    return jaccard_score
