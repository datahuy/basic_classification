import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.backends.cudnn.enabled = True


class Encoder(nn.Module):
    def __init__(self, label_size, hidden_size, vocab_size, dropout=0.3):
        """
        Define a basic neural network with these components:
        - layer 1: a fully connected layer with the activation function is relu
        - layer 2: a fully connected layer converts layer 1 output for each sentences to a distribution over labels
        
        Args:
            params: (Params) contains vocab_size, ... 
        """
        super(Encoder, self).__init__()
        self.label_size = label_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.dropout = nn.Dropout(dropout)
        self.layer1 = nn.Parameter(torch.FloatTensor(hidden_size, vocab_size))
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

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all setences..

    Args:
        outputs: (np.ndarray) dimension batch_size x num_labels - output of the model
        labels: (np.ndarray) dimension batch_size x 1 where each element is either a label in
                [0, 1, ... num_labels-1].

    Returns: (float) accuracy in [0,1]
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels)/float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
