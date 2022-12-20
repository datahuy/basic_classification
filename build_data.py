import argparse
import os

from sklearn.model_selection import train_test_split

import utils
from main.utils.preprocess_text.preproces_industry_cls import clean_text

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data_demo', help="Directory containing the dataset")
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
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences:
        file_sentences.write("\n".join(data['sentences']))
    
    with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
        file_labels.write("\n".join(data['labels']))
    print("- Done!")


if __name__ == "__main__":
    args = parser.parse_args()

    path_dataset = os.path.join(args.data_dir, args.data_name)
    msg = "{} file not found. Make sure you have downloaded \
        the right dataset".format(path_dataset)
    assert os.path.isfile(path_dataset), msg

    # Load the dataset into memory
    print("\n\nLoading csv dataset into memory...")
    raw_data = utils.load_csv(
        csv_file=path_dataset,
        col_sentences='sentences',
        col_labels='labels',
        sep=','
    )
    print("- Done!")

    raw_data['sentences'] = list(map(clean_text, raw_data['sentences']))

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
    save_dataset(val_data, os.path.join(args.data_dir, 'test'))
