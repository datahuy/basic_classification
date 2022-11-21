"""Evaluates the model"""

import argparse
import os

import torch
from tqdm import tqdm

import utils
from model.product_classify import ProductClassify
from preprocessing import clean_text

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kiot-viet', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def infer(sentences, threshold=0.75, return_probs=False, model_dir='experiments/base_model', data_dir="data/kiot-viet"):
    """Infer labels respectively with sentences

    Args:
        sentences: (nd.array or list) sentences input
        model_dir: directory containing model, params,...
        data_dir: directory containing data train, vocab, ...
        threshold: (float) Default is 0.75. If probability is less than thredshold, return industry "unknown"
        return_prob: (bool) Default is False. If true, return both list contain labels and probabilities respectively

    Return:
        List containing labels reeespectively with sentences

    Examples
    ----------
    Infer list of product names

    >>> infer(['iPhone 14 promax'])
    ['Điện tử - Điện máy']

    Adjust threshold to trade-off precision-recall

    >>> infer(['lồng đèn búp bê '], return_probs=True)
    [('Mẹ & Bé', 0.74538445)]

    >>> infer(['lồng đèn búp bê '], threshold=0.7)
    ['Mẹ & Bé']

    >>> infer(['lồng đèn búp bê '], threshold=0.8)

    ['unknown']   
    
    """

    print("Loading checkpoint ...")
    # Params containing config of model 
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path=json_path)

    # Append dataset_params 
    json_path = os.path.join(data_dir, 'dataset_params.json')
    assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
    params.update(json_path=json_path)

     # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(100)
    if params.cuda: torch.cuda.manual_seed(100)

    # Loading vocab, map words to their indices
    vocab_path = os.path.join(data_dir, 'words.txt')
    vocab = {}
    with open(vocab_path) as fi:
        for index, word in enumerate(fi.read().splitlines()):
            vocab[word] = index
    unkown_index = vocab[params.unk_word]

    # Loading labels, create map from label indices to label
    labels_path = os.path.join(data_dir, 'labels.txt')
    label_decode = {}
    with open(labels_path) as fi:
        for index, label in enumerate(fi.read().splitlines()):
            label_decode[index] = label

    # Load file checkpoint
    model = ProductClassify(params=params)
    utils.load_checkpoint(
        checkpoint=os.path.join(model_dir, 'best.pth.tar'),
        model=model
    )
    print(" - Done !")


    # Preprocessing sand indexing sentences:
    clean_sentences = [
        clean_text(sentence) for sentence in tqdm(sentences, desc="Preprocessing")
    ]
    sentences_indices = [
        [
            vocab[token] if token in vocab else unkown_index
            for token in sentence.split()
        ]
        for sentence in clean_sentences
    ]
    size_data = len(sentences_indices)
    labels = [0] * size_data 

    # Initialize json data before parsing to model
    data = {
        "sentences" : sentences_indices,
        "labels" : labels,
        "size": size_data
    }
    print(" - Done!")
    labels_indices, labels_probs = model.evaluate(
        data=data,
        infer=True        
    )

    # Map from indices to label respectively
    labels = [
        label_decode[label_indices] for label_indices in labels_indices
    ]

    # Assign to unknown if probability prediction is less than threshold
    labels = [
        label if prob >= threshold else 'unknown'
        for label, prob in list(zip(labels, labels_probs))
    ]
    # Round probabilities
    labels_probs = [round(prob, 2) for prob in labels_probs]

    print(" - Done!")

    if return_probs:
        return list(zip(labels,labels_probs))
    return labels


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    res = infer(
        sentences=['iphone 14 pro max', 'bimbim oishi', 'Pink Midi Dress'],
        model_dir='experiments/base_model',
        data_dir='data/kiot-viet'
    )
    print(res)