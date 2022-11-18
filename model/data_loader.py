from email import utils
from itertools import chain
import os
import random

import numpy as np
import torch
from torch.autograd import Variable

import utils


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and labels with
    their mappings to indices
    """
    def __init__(self, data_dir, params) -> None:
        """
        Loading dataset_params, vocabulary and labels. Ensure running 'build_vocab.py on 
        data_dir before using this class.
        
        Args:
            data_dir: (string) directory containing the dataset.
            params: (Params) hyperparameters of the training process.
                    This function modifies params and append dataset_params to params.

        """

        # Loading dataset_params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        
        self.dataset_params = utils.Params(json_path=json_path)

        # Loading vocab, map words to their indices
        vocab_path = os.path.join(data_dir, 'words.txt')
        self.vocab = {}
        with open(vocab_path) as fi:
            for index, word in enumerate(fi.read().splitlines()):
                self.vocab[word] = index
        # Setting the indices for unknown words
        self.unk_ind = self.vocab[self.dataset_params.unk_word]
        
        # Loading labels, map them to their indices
        labels_path = os.path.join(data_dir, 'labels.txt')
        self.label_map = {}
        with open(labels_path) as fi:
            for index, label in enumerate(fi.read().splitlines()):
                self.label_map[label] = index
        
        # Adding dataset parameters to param (e.g. vocab size, etc.)
        params.update(json_path)

    def load_sentences_labels(self, sentences_file, labels_file, d):
        """
        Loads sentences and labels from their corresponding files. Maps tokens and tags to their indices and stores
        them in the provided dict d.

        Args:
            sentences_file: (string) file with sentences with tokens space-separated
            labels_file: (string) file with NER tags for the sentences in labels_file
            d: (dict) a dictionary in which the loaded data is stored
        """

        sentences = []
        labels = []

        with open(sentences_file) as fi:
            for sentence in fi.read().splitlines():
                """ Replace each token by its index if it is in vocab
                    else use index of UNK_WORD """
                s = [self.vocab[token] if token in self.vocab
                     else self.unk_ind
                     for token in sentence.split(' ')]
                sentences.append(s)
        
        with open(labels_file) as fi:
            for label in fi.read().splitlines():
                labels.append(self.label_map[label])

        # Check to ensure there is a label for each sentence
        assert len(labels) == len(sentences)

        # Storing sentences and labels in dict d
        d['sentences'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):
        """
        Loads the data for each type in types from data_dir

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on
                    which data is required.
            data_dir: (string) directory containing the dataset
        Returns:
            data: (dict) contains the data with labels for each type in types 
        """
        data = {}
        
        for split in ['train', 'val', 'test']:
            if split in types: 
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}
                self.load_sentences_labels(
                    sentences_file=sentences_file,
                    labels_file=labels_file,
                    d=data[split]
                )
    
        return data
    
    @staticmethod
    def data_iterator(data, params, shuffle=False):
        """
        Returns a generator that yeilds batches data with labels. Batch size is params.batch_size.
        Expires after one pass over the data.
        
        Args:
            data: (dict) contains data which has keys 'sentences', 'labels' and 'size'.
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled.

        Yiels:
            batch_sentences: (Variable) dimension batch_size x seq_len with the sentences data
            bath_labels: (Variable) dimension batch_size x seq_len with the sentences data with the corresponding labels

        """

        # Make a list that decides the order in which we go over the data - this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(100)
            random.shuffle(order)

        # One pass over data
        for i in range((data['size'] + 1) // params.batch_size):
            
            # Fetch sentences and labels:
            batch_indices_sentences = [data['sentences'][idx] for idx in order[i*params.batch_size : (i+1)*params.batch_size]]
            batch_indices_labels = [data['labels'][idx] for idx in order[i*params.batch_size : (i+1)*params.batch_size]]

            # Embedding BOW from sentence indices, convert to torch.LongTensors
            batch_embedding_sentences = np.zeros((len(batch_indices_sentences), params.vocab_size))
            for i in range(len(batch_embedding_sentences)):
                batch_embedding_sentences[i][batch_indices_sentences[i]] = 1
            
            batch_embedding_sentences = torch.tensor(batch_embedding_sentences, dtype=torch.float32)
            batch_indices_labels = torch.LongTensor(batch_indices_labels)
            # Shift tensors to GPU if available
            if params.cuda:
                batch_embedding_sentences, batch_indices_labels = batch_embedding_sentences.cuda(), batch_indices_labels.cuda()
            
            # Convert them to Variables to record operations in the computational graph
            batch_embedding_sentences = Variable(batch_embedding_sentences)
            batch_indices_labels = Variable(batch_indices_labels)

            yield batch_embedding_sentences, batch_indices_labels