from utils import (
    get_data,
    get_urls_parsed,
    get_labels,
    get_data_splitted,
    convert_text_to_sequences,
    save_tokenizer,
    save_labelbinarizer,
    save_loss_plt,
    save_acc_plt
)
from config.config import (
    DATA_TYPE,
    MAX_LEN,
    RANDOM_STATE,
    TEST_SIZE,
    TRAINED_MODEL_OUTPUT_PATH,
    EPOCHS,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    TOKENIZER_OUTPUT_PATH,
    LABELBINARIZER_OUTPUT_PATH,
    MODEL_NAME,
    PLT_IMAGES_PATH
)
from model import train_model, evaluate_model, get_jaccard_score
import argparse

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Multi-label URL classification')

    # Add the argument
    parser.add_argument('--path',
                        dest='path',
                        type=str,
                        help='the data path')

    args = parser.parse_args()

    # Get the data path
    DATA_PATH = args.path

    # Get the data
    df = get_data(data_path=DATA_PATH, data_type='csv')

    # Get the parsed url
    all_url_parsed = get_urls_parsed(df=df)

    # Binarize the labels with MultiLabelBinarizer
    mlb, all_binary_labels = get_labels(df)

    # Get data splitted
    train_data, test_data, y_train, y_test = get_data_splitted(all_url_parsed,
                                                               all_binary_labels,
                                                               random_state=RANDOM_STATE,
                                                               test_size=TEST_SIZE)

    # Get the text vectors
    X_train, X_test, vocab_size, tokenizer = convert_text_to_sequences(all_text_url=all_url_parsed,
                                                                       max_len=MAX_LEN,
                                                                       sentences_train=train_data,
                                                                       sentences_test=test_data)
    # Training
    history = train_model(x=X_train,
                          y=y_train,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          validation_split=VALIDATION_SPLIT,
                          vocab_size=vocab_size,
                          max_len=MAX_LEN,
                          trained_model_output_path=TRAINED_MODEL_OUTPUT_PATH,
                          model_name=MODEL_NAME)

    # Save the tokenizer
    save_tokenizer(tokenizer=tokenizer,
                   tokenizer_output_path=TOKENIZER_OUTPUT_PATH)

    # Save MultiLabelBinarizer
    save_labelbinarizer(mlb=mlb,
                        labelbinarizer_output_path=LABELBINARIZER_OUTPUT_PATH)

    # Save plt images
    loss_plt = save_loss_plt(history=history, plt_images_path=PLT_IMAGES_PATH)
    acc_plt = save_acc_plt(history=history, plt_images_path=PLT_IMAGES_PATH)

    # Test the model (metrics : acc)
    results = evaluate_model(history=history, x_test=X_test, y_test=y_test)

    # Get Jaccard score
    jaccard_score = get_jaccard_score(history=history, x_test=X_test, y_test=y_test)
    
    
