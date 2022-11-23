import math

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from math import ceil
from tqdm import tqdm
import os
import time


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all setences..

    Args:
        outputs: (np.ndarray) dimension batch_size x num_labels - output of the model
        labels: (np.ndarray) dimension batch_size x 1 where each element is either a label in
                [0, 1, ... num_labels-1].

    Returns: (float) accuracy in [0,1]
    """
    labels = labels.ravel()
    mask = (labels >= 0)
    outputs = np.argmax(outputs, axis=1)

    return np.sum(outputs == labels)/float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}



def load_model(model_file, use_gpu=False):
    # load the pre-trained model
    with open(model_file, 'rb') as f:
        # If we want to use GPU and CUDA is correctly installed
        if use_gpu and torch.cuda.is_available():
            state = torch.load(f)
        else:
            # Load all tensors onto the CPU
            state = torch.load(f, map_location='cpu')
    if not state:
        return None
    vocab = state['vocab']
    label2index = state['label2index']
    hidden_size = state['hidden_size']
    vocab_size = len(vocab)
    label_size = len(label2index)

    index2label = {v: k for k, v in label2index.items()}

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
        start = time.time()
        layer1_output = F.relu(F.linear(inps, self.layer1, self.b1))
        layer1_output = self.dropout(layer1_output)
        logit = F.linear(layer1_output, self.layer2, self.b2)
        return logit


class ModelClassifer():
    def __init__(self, vocab, label2index):
        """ Initialize paramters and components for training process

        Args:
            params: (Params) store parameter for model
        """
        # Initialize the parameters
        super(ModelClassifer, self).__init__()
        self.vocab = vocab
        self.label2index = label2index
        # self.params = params

        # Define the encoder
        self.encoder = Encoder(
            label_size=5, #params.labels_size,
            hidden_size= 200, #params.hidden_size,
            max_words= 53957,#params.vocab_size,
            dropout=0.3, #params.dropout
        )

    def predict(self):
        print('predict')

    def predict_batch(self):
        print('predict_batch')

def collate_fn(data):
    text, label = zip(*data)
    
    tokens = list(map(lambda x: [vocab_dict[w] for w in x.split(' ')], list(text)))
    text_embeddings = torch.zeros(len(text), len(vocab_dict))
    for idx, item in enumerate(tokens):
        item = torch.LongTensor(item)
        text_embeddings[idx].index_add_(0, item, torch.ones(item.size()))

    return text_embeddings, torch.LongTensor(label)

class ClassifierTrainer():
    def __init__(self, 
                model,
                train_dataset,
                eval_dataset,
                model_path):
        self.model_path = model_path
        self.model = model
        # self.params = params
        self.num_epochs = 10 #params.num_epochs
        self.train_batch_size = 128 #params.train_batch_size
        self.eval_batch_size = 128 #params.eval_batch_size
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size, collate_fn=collate_fn)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=self.eval_batch_size, collate_fn=collate_fn)
        self.encoder = model.encoder
        self.encoder.cuda()
        self.train_dataset = train_dataset

    def train(self):
        self.optimizer = optim.Adam(
            params=self.encoder.parameters(),
            lr=0.001#self.params.learning_rate
        )
        self.loss_function = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        logging.info("Starting training for {} epoch(s)".format(self.num_epochs))
        for epoch in range(self.num_epochs):
            # Run one epoch
            print(epoch)
            logging.info("Epoch {}/{}".format(epoch + 1, self.num_epochs))

            """Training process"""
            # Set model to training mode            
            self.encoder.train()

            # Summary for current training loop and a running average object for loss
            
            losses = []
            start = time.time()
            for i, (embedding_text, labels) in tqdm(enumerate(self.train_dataloader)):
                # Compute model output and loss
                self.encoder.zero_grad()

                outputs = self.encoder(embedding_text.to('cuda'))
                loss = self.loss_function(outputs, labels.to('cuda'))
            
                # Compute gradients of loss function with all variables
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
            
                # Performs updates using calculated gradients
                self.optimizer.step()

                # Evaluate summaries only once in a while
                # if i % self.params.save_summary_steps == 0:
                    
                #     # Extract data from torch Variable, move to cpu, convert to numpy arrays
                #     outputs = outputs.data.cpu().numpy()
                #     labels = labels.data.cpu().numpy()
                    
                #     # Compute metrics on this batch
                #     summary_batch = {
                #         metric: metrics[metric](outputs, labels)
                #         for metric in metrics
                #     }
                #     summary_batch['loss'] = loss.item()
                #     summ.append(summary_batch)

                # Update the average loss
                losses.append(loss.cpu().detach().numpy())
            print(np.mean(np.array(losses)))
            # Compute mean of all metrics in summary
            # metrics_mean = {
            #     metric: np.mean([x[metric] for x in summ])
            #     for metric in summ[0]
            # }
            # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
            #                              for k, v in metrics_mean.items())
            # logging.info("- Train metrics: " + metrics_string)            


            # """Validate process"""
            # val_metrics, val_metrics_string = self.evaluate()
            # logging.info("- Eval metrics : " + val_metrics_string)


            # val_acc = val_metrics['accuracy']
            # is_best = val_acc >= best_val_acc

            # # Save weights
            # self.save_model(self.model_path)

            # # If best_eval, best_save_path
            # if is_best:
            #     logging.info("- Found new best accuracy!")
            #     best_val_acc = val_acc
                
            #     # Save best val metrics in a json file in the model directory
            #     best_json_path = os.path.join(
            #         self.model_path, "metrics_val_best_weights.json"
            #     )
            #     utils.save_dict_to_json(val_metrics, best_json_path)        

            # # Save latest val metrics in a json file in the model directory
            # last_json_path = os.path.join(
            #     self.model_path, "metrics_val_last_weights.json"
            # )
            # utils.save_dict_to_json(val_metrics, last_json_path)


    # def evaluate(self):
    #     self.encoder.eval()

    #     # Compute metrics over the dataset
    #     for i, (text_embeddings, labels) in enumerate(self.eval_dataloader):
    #         # Compute model output
    #         outputs = self.encoder(text_embeddings)
    #         loss = self.loss_function(outputs, labels)
    #         probs = torch.softmax(outputs, dim=1)

    #         # Extract data from torch Variable, move to cpu, convert to numpy arrays
    #         output_batch = output_batch.data.cpu().numpy()
    #         labels_batch = labels_batch.data.cpu().numpy()
    #         probs = probs.data.cpu().numpy()

    #         # Compute all metrics on this batch
    #         summary_batch = {
    #             metric: metrics[metric](output_batch, labels_batch)
    #             for metric in metrics
    #         }
    #         summary_batch['loss'] = loss.item()
    #         summ.append(summary_batch)
        
    #     # Compute mean of all metrics in summary
    #     metrics_mean = {
    #         metric: np.mean([x[metric] for x in summ])
    #         for metric in summ[0]
    #     }
    #     metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())

    #     return metrics_mean, metrics_string



    def save_model(self, model_path):
        state = {
            'state_dict': self.encoder.state_dict(),
            'vocab': self.model.vocab,
            'label2index': self.model.label2index,
            'hidden_size': self.params.hidden_size,
        }
        with open(model_path, 'wb') as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    from data.classifier_dataset import ClassifierDataset
    data_dir = '/root/classify_data'
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        labels = f.read().splitlines()
    labels = list(set(labels))
    label2index = {}
    for i in range(len(labels)):
        label2index[labels[i]] = i
    
    with open(os.path.join(data_dir, 'words.txt'), 'r') as f:
        vocab = f.read().splitlines()
    vocab_dict = {}
    for i in range(len(vocab)):
        vocab_dict[vocab[i]] = i

    train_dataset = ClassifierDataset(data_dir, 'train', vocab_dict, label2index)
    eval_dataset = ClassifierDataset(data_dir, 'test', vocab_dict, label2index)

    model = ModelClassifer(vocab, label2index)
    trainer = ClassifierTrainer(model, train_dataset, eval_dataset, './')
    trainer.train()

    #{'unknown': 0, 'Mỹ phẩm': 1, 'Thời trang': 2, 'Mẹ & Bé': 3, 'Điện tử - Điện máy': 4}