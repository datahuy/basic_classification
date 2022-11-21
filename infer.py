"""Evaluates the model"""

import argparse
from cProfile import label
import logging
import os
from tabnanny import check

import numpy as np
import torch
import utils
from model.product_classify import ProductClassify
from model.data_loader import DataLoader
from preprocessing import clean_text

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kiot-viet', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")

def infer(sentences, model_dir, data_dir):
    """Infer labels respectively with sentences

    Args:
        sentences: (nd.array or list) sentences input
        model_dir: directory containing model, params,...
        data_dir: directory containing data train, vocab, ...
    
    Return:
        List containing labels reeespectively with sentences
    """

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
    
    # Preprocessing sand indexing sentences:
    clean_sentences = [
        clean_text(sentence) for sentence in sentences
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

    # Load file checkpoint
    model = ProductClassify(params=params)
    utils.load_checkpoint(
        checkpoint=os.path.join(model_dir, 'best.pth.tar'),
        model=model
    )

    labels_indices = model.evaluate(
        data=data,
        infer=True        
    )

    # Map from indices to label respectively
    labels = [
        label_decode[label_indices] for label_indices in labels_indices
    ]

    return labels



if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    # args = parser.parse_args()
    # json_path = os.path.join(args.model_dir, 'params.json')
    # assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # params = utils.Params(json_path)

    # # use GPU if available
    # params.cuda = torch.cuda.is_available()     # use GPU is available

    # # Set the random seed for reproducible experiments
    # torch.manual_seed(100)
    # if params.cuda: torch.cuda.manual_seed(100)
        
    # # Get the logger
    # utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # # Create the input data pipeline
    # logging.info("Creating the dataset...")

    # # load data
    # data_loader = DataLoader(args.data_dir, params)
    # data = data_loader.load_data(['test'], args.data_dir)
    # test_data = data['test']

    # logging.info("- done.")

    # # Define the model
    # model = ProductClassify(params=params)
    
    # logging.info("Starting evaluation")

    # # Reload weights from the saved file
    # # utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    # utils.load_checkpoint(
    #     checkpoint=os.path.join(args.model_dir, args.restore_file + '.pth.tar'),
    #     model=model
    # )

    # Evaluate
    # num_steps = (params.test_size + 1) // params.batch_size
    # test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
    # save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    # utils.save_dict_to_json(test_metrics, save_path)

    labels = infer(
        sentences=1000 * ['túi hình poli', 'mặt_nạ lanybeau vàng', 'bỉm yubest gold quần', 'iphone x bao da loai tot k', 'củ sạc dẹt iphone original'],
        model_dir='experiments/base_model',
        data_dir='data/kiot-viet'
    )
    print(labels)