"""Build vocabularies of words and labels from datasets"""

import argparse
import json
import os
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=1, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")

# Hyper parameters for the vocab
PAD_WORD = '<pad>'
UNK_WORD = 'UNK'


def update_vocab(txt_path, vocab):
    """Update word and tag vocabulary from dataset

    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method

    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path, encoding='utf-8') as fi:
        for i, line in enumerate(fi):
            vocab.update(line.strip().split(' '))

    return i + 1


def save_vocab(vocab, txt_path):
    """Writes one token per line, 0-based line id corresponds to the id of the token.

    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w", encoding='utf-8') as fo:
        for token in vocab:
            fo.write(token + '\n')


def save_dict_to_json(d: dict, json_path):
    """Saves dict to json file

    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as fo:
        d = {k: v for k, v in d.items()}
        json.dump(d, fo, indent=4)


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train dataset
    print("Build word vocab with train dataset")
    words = Counter()
    size_train_sentences = update_vocab(
        txt_path=os.path.join(args.data_dir, 'train/sentences.txt'),
        vocab=words
    )
    print("- Done.")
    
    # Build labels vocab with train dataset
    print("Build labels vocabulary...")
    with open(os.path.join(args.data_dir, 'train/labels.txt'), encoding = 'utf-8') as fi:
        labels = list(set(fi.read().splitlines()))
    print("- Done!")

    # Only keep most frequent tokens
    words = [tok for tok, count in words.items() if count >= args.min_count_word]

    # add word for unknown words 
    words.append(UNK_WORD)

    # Save vocabularies to file
    print("Saving vocabularies to file...")
    save_vocab(words, os.path.join(args.data_dir, 'words.txt'))
    save_vocab(labels, os.path.join(args.data_dir, 'labels.txt'))
    print("- done.")

    # Save dataset's properties into json file
    properties = {
        'train_size': size_train_sentences,
        'vocab_size': len(words),
        'labels_size': len(labels),
        'unk_word': UNK_WORD,
    }

    save_dict_to_json(properties, os.path.join(args.data_dir, 'dataset_params.json'))

    # Logging dataset's properties
    to_print = "\n".join("- {}: {}".format(k, v) for k, v in properties.items())
    print("Information of the dataset:\n{}".format(to_print))