import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import pandas as pd
from glob import glob
from tqdm import tqdm
import string
import re
from urllib.parse import unquote
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt

import logging
logging.basicConfig(filename='training_log.log', level=logging.DEBUG)


def get_data(data_path: str, data_type: str):
    """
    To get the data into dataframe.

    :param data_type: type of data (csv, parquet etc.)
    :param data_path: data path
    :return: Dataframe concatenated
    """
    name = ["url", "target"]
    data = [pd.read_csv(data_path, names=name)]

    logging.info('Data uploaded !')

    return pd.concat(data, ignore_index=True)


def clean_and_parse_url(url: str) -> str:
    """
    To clean and parse the url with regex.

    :param url: Url
    :return: cleaned and parsed url
    """
    # exclude the punctuations
    exclude = set(string.punctuation)

    # we split the url into protocol, domain name and URI.
    # Because, we prefer to do the pre-processing separately
    pattern = re.compile(r'^(?P<protocol>.*?)://(?P<domainname>.*?)(/(?P<uri>.*?))?$')
    groups = re.match(pattern, url)
    if groups is not None:
        groups = groups.groupdict()
    # DOMAIN NAME PRE-PROCESSING
    # delete the protocol "www."
        groups["domainname"] = re.sub(r"www.", "", groups["domainname"])

        # Remove punctuation from domain name
        groups["domainname"] = ''.join(ch if ch not in exclude else ' ' for ch in groups["domainname"])

        # URI PRE-PREPROCESSİNG
        if groups["uri"]:
            # remove punctuations
            groups["uri"] = ''.join(ch if ch not in exclude else ' ' for ch in groups["uri"])
            # remove html or htm tag
            groups["uri"] = re.sub(r".html?$", "", groups["uri"])
            # remove digits
            groups["uri"] = re.sub(r"\d", "", groups["uri"])
            # remove some whitespaces
            groups["uri"] = re.sub(r"\s{2,}", " ", groups["uri"])
            # remove token if the token's length is less than 2
            cleaned_token = [token for token in groups["uri"].split() if not len(token) < 2]
            groups["uri"] = " ".join(cleaned_token)

        # We take only domain name and URI and concat them
        text_concatenated = f"{groups['domainname']} {groups['uri']}".lower()

        return text_concatenated
    else: 
        return url


def get_urls_parsed(df) -> list:
    """
    To get all the url parsed.

    :param df: dataframe
    :return: list of parsed and converted url (to str)
    """
    all_url_parsed = list()
    url_data = df.url.tolist()

    print("Data preprocessing...")
    for url in tqdm(url_data, total=len(url_data)):
        url_parsed = clean_and_parse_url(url)
        all_url_parsed.append(url_parsed)

    return all_url_parsed


def get_labels(df):
    """
    To get the binary labels that can be used in the model.

    :param df: dataframe
    :return: multilabelbinarizer object, all binary labels array
    """
    mlb = MultiLabelBinarizer()
    all_labels_binarized = mlb.fit_transform(df.target.values.tolist())

    logging.info('MultiLabelBinarized uploaded !')

    return mlb, all_labels_binarized


def get_data_splitted(all_text, all_label, random_state, test_size):
    """
    Split the data for training and testing.

    :param all_text: all sentences in the dataframe
    :param all_label: all labels in the dataframe
    :param random_state: random state
    :param test_size: test size defined
    :return: splitted data (X_train, X_test, y_train, y_test)
    """
    x_train, x_test, y_train, y_test = train_test_split(all_text,
                                                        all_label,
                                                        random_state=random_state,
                                                        test_size=test_size
                                                        )

    logging.info(f'Train test split completed !')

    return x_train, x_test, y_train, y_test


def convert_text_to_sequences(all_text_url, sentences_train, sentences_test, max_len):
    """
    Transforms text data to feature_vectors that can be used in the model.

    :param all_text_url: all sentences in dataframe
    :param sentences_train: train sentences
    :param sentences_test: test sentences
    :param max_len: maximum length for padding
    :return: vectors for x_train, x_test and total vocabulary size number
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_text_url)

    x_train = tokenizer.texts_to_sequences(sentences_train)
    x_test = tokenizer.texts_to_sequences(sentences_test)

    x_train = pad_sequences(x_train, padding='post', maxlen=max_len)
    x_test = pad_sequences(x_test, padding='post', maxlen=max_len)

    vocab_size = len(tokenizer.word_index) + 1

    logging.info('Texts converted into sequences and padded !')

    return x_train, x_test, vocab_size, tokenizer


def save_tokenizer(tokenizer, tokenizer_output_path):
    """
    To save the tokenizer.

    :param tokenizer: tokenizer
    :param tokenizer_output_path: tokenizer output path
    :return: None
    """
    # saving
    with open(tokenizer_output_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Tokenizer is successfully saved.")


def save_labelbinarizer(mlb, labelbinarizer_output_path):
    """
    To save MultiLabelBinarizer.

    :param mlb: MultiLabelBinarizer
    :param labelbinarizer_output_path: output path
    :return: None
    """
    # saving
    with open(labelbinarizer_output_path, 'wb') as handle:
        pickle.dump(mlb, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("MutliLabelBinarizer is saved.")


def save_loss_plt(history, plt_images_path):
    """
    To draw a loss curve.

    :param history: model history to get the loss score
    :param plt_images_path: plot images output path
    :return: None
    """
    # Get training and test loss histories
    training_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, val_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{plt_images_path}/loss.png')
    plt.show();

    logging.info('Loss plot image saved !')


def save_acc_plt(history, plt_images_path):
    """
    To draw an acc curve.

    :param history: model history to get the accuracy score
    :param plt_images_path: plot images output path
    :return: None
    """
    # Get training and test acc histories
    training_acc = history.history['acc']
    val_acc = history.history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_acc) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_acc, 'r--')
    plt.plot(epoch_count, val_acc, 'b-')
    plt.legend(['Training Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig(f'{plt_images_path}/acc.png')
    plt.show();

    logging.info('Accuracy plot image saved !')

