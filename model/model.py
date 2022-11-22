import math

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import logging



def load_model(model_file, use_gpu=False):
    # load the pre-trained model
    with open(model_file, 'rb') as f:
        # If we want to use GPU and CUDA is correctly installed
        if use_gpu and torch.cuda.is_available():
            state = torch.load(f)
        else:
            # Load all tensors onto the CPU
            state = torch.load(f, map_location='cpu')
    logger.debug(f'model state = {str(state)}')
    if not state:
        return None
    vocab = state['vocab']
    label2index = state['label2index']
    hidden_size = state['hidden_size']
    vocab_size = len(vocab)
    label_size = len(label2index)

    index2label = {v: k for k, v in self.label2index.items()}

    params = {
        'vocab': vocab,
        'label_size': label_size,
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'label2index': label2index,
        'index2label': index2label
    }
    model_classifier = ModelClassfier(params)
    model_classifier.encoder.load_state_dict(state['state_dict'])

    if use_gpu:
        model_classifier.encoder.cuda()
    return model_classifier


class Encoder(nn.Module):
    def __init__(self, label_size, hidden_size, max_words, dropout=0.3):
        super(Encoder, self).__init__()
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.max_words = max_words

        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Parameter(torch.FloatTensor(hidden_size, max_words))
        self.b1 = nn.Parameter(torch.FloatTensor(hidden_size))
        self.layer2 = nn.Parameter(torch.FloatTensor(label_size, hidden_size))
        self.b2 = nn.Parameter(torch.FloatTensor(label_size))

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters
        :return:
        """
        stdv1 = 1.0 / math.sqrt(self.hidden_size)
        self.layer1.data.uniform_(-stdv1, stdv1)
        self.b1.data.uniform_(-stdv1, stdv1)

        stdv2 = 1.0 / math.sqrt(self.label_size)
        self.layer2.data.uniform_(-stdv2, stdv2)
        self.b2.data.uniform_(-stdv2, stdv2)

    def forward(self, inps):
        layer1_output = F.relu(F.linear(inps, self.layer1, self.b1))
        layer1_output = self.dropout(layer1_output)
        logit = F.linear(layer1_output, self.layer2, self.b2)

        return logit


class ModelClassifer():
    def __init__(self, params, vocab, label2index):
        """ Initialize paramters and components for training process

        Args:
            params: (Params) store parameter for model
        """
        # Initialize the parameters
        super(ModelClassifer, self).__init__()
        self.vocab = vocab
        self.label2index = label2index
        self.params = params

        # Define the encoder
        self.encoder = Encoder(
            label_size=params.labels_size,
            hidden_size=params.hidden_size,
            vocab_size=params.vocab_size,
            dropout=params.dropout
        )

    def predict(self):
        pass

    def predict_batch(self):
        pass


class ClassifierTrainer():
    def __init__(self, 
                model: ModelClassifier,
                params,
                train_dataset,
                eval_dataset):
        self.model = model
        self.params = params
        self.train_dataloader = train_dataset
        self.eval_dataloader = eval_dataset

    def train():
        self.optimizer = optim.Adam(
            params=self.encoder.parameters(),
            lr=params.learning_rate
        )
        self.loss_function = nn.CrossEntropyLoss()

    def evaluate():
        pass

    def save_model(self, model_path):
        state = {
            'state_dict': self.model.encoder.state_dict(),
            'vocab': self.model.vocab,
            'label2index': self.model.label2index,
            'hidden_size': self.params.hidden_size,
        }
        with open(model_path, 'wb') as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)
