import argparse
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)

def load_csv(csv_file, col_sentences, col_labels, sep):
    """Load csv file, convert data to dictionary

    Args:
        csv_file: path to csv data
        col_sentences: column containing sentences
        col_labels: column containing labels
        sep: delimiter between 2 columns
    """

    df = pd.read_csv(csv_file, sep=sep)
    df = df.astype(str)
    # Remove new line character in sentences
    df[col_sentences] = df[col_sentences].str.replace('\n', ' ').str.replace('\r', ' ').str.strip()
    data = {
        "sentences": list(df[col_sentences].values),
        "labels": list(df[col_labels].values)
    }           
    # logging.info(f"data: {data}")
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
parser.add_argument('--data_name', default='product.csv', help='Name of file data')


def save_dataset(data: dict, save_dir):
    """Writes sentences.txt and labels.txt files in save_dir from dataset

    Args:
        dataset: [("iphnone 14 promax", "Điện tử - Điện máy"), ...]
        save_dir: (string)
    """

    # Create directory if it doesn't exist
    print("Saving in {}...".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w', encoding='utf-8') as file_sentences:
        file_sentences.write("\n".join(data['sentences']))
    logging.info("Number of sentences: {}".format(len(data['sentences'])))
    
    with open(os.path.join(save_dir, 'labels.txt'), 'w', encoding='utf-8') as file_labels:
        file_labels.write("\n".join(data['labels']))
    logging.info("Number of labels: {}".format(len(data['labels'])))
    print("- Done!")


if __name__ == "__main__":
    args = parser.parse_args()

    path_dataset = os.path.join(args.data_dir, args.data_name)
    msg = "{} file not found. Make sure you have downloaded \
        the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("\n\nLoading csv dataset into memory...")
    raw_data = load_csv(
        csv_file=path_dataset,
        col_sentences='sentences',
        col_labels='labels',
        sep=','
    )
    print("- Done!")

    """Split the dataset into train, val and test
    (dummy split with no shuffle)"""
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(
        raw_data['sentences'],
        raw_data['labels'],
        test_size=0.2,
        random_state=100
    )

    val_sentences, test_sentences, val_labels, test_labels = train_test_split(
        test_sentences,
        test_labels,
        test_size=0.5,
        random_state=100
    )


    train_data = {
        'sentences': train_sentences,
        'labels': train_labels
    }
    val_data = {
        'sentences': val_sentences,
        'labels': val_labels
    }
    test_data = {
        'sentences': test_sentences,
        'labels': test_labels
    }

    # Save the datasets to files
    save_dataset(train_data, os.path.join(args.data_dir, 'train'))
    save_dataset(val_data, os.path.join(args.data_dir, 'val'))
    save_dataset(test_data, os.path.join(args.data_dir, 'test'))