import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.label_size = config['label_size']
        self.hidden_size = config['hidden_size']
        self.max_words = config['max_words']
        self.dropout_prob = config['dropout']

        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.max_words))
        self.b1 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        self.layer2 = nn.Parameter(torch.FloatTensor(self.label_size, self.hidden_size))
        self.b2 = nn.Parameter(torch.FloatTensor(self.label_size))

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


class ClassfierModel():
    def __init__(self, vocab_dict, index2label, config):
        """ Initialize paramters and components for training process

        Args:
            params: (Params) store parameter for model
        """
        # Initialize the parameters
        self.vocab_dict = vocab_dict
        self.index2label = index2label

        # Define the encoder
        config['label_size'] = len(index2label)
        config['max_words'] = len(vocab_dict)
        self.encoder = Encoder(
            config
        )

    def preprocess(self, input):
        tokens = list(map(lambda x: [self.vocab_dict[w] if w in self.vocab_dict else self.vocab_dict['UNK'] for w in
                                     x.split(' ')], input))
        text_embeddings = torch.zeros(len(input), len(self.vocab_dict))
        for idx, item in enumerate(tokens):
            item = torch.LongTensor(item)
            text_embeddings[idx].index_add_(0, item, torch.ones(item.size()))
        return text_embeddings

    def predict(self, input: List, batch_size):
        text_embeddings = self.preprocess(input)
        probs = []
        preds = []
        with torch.no_grad():
            for i in range(0, len(input), batch_size):
                outputs = self.encoder(text_embeddings[i:i + batch_size])
                outputs = torch.softmax(outputs, dim=-1)
                cur_probs, cur_preds = torch.max(outputs, dim=-1)
                preds.extend([self.index2label[p] for p in cur_preds.tolist()])
                probs.extend(cur_probs.tolist())
        return preds, probs
