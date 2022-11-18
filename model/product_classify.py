import logging
import os

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import model.encoder as encoder
import utils
from model.data_loader import DataLoader




torch.backends.cudnn.enabled = True

logger = logging.getLogger()
metrics = encoder.metrics

class ProductClassify(nn.Module):
    def __init__(self, params):
        """ Initialize paramters and components for training process

        Args:
            params: (Params) store parameter for model
        """
        # Initialize the parameters
        super(ProductClassify, self).__init__()
        # self.num_epochs = params.num_epochs
        # self.batch_size = params.batch_size
        # self.hidden_size = params.hidden_size
        # self.dropout = params.dropout
        # self.learning_rate = params.learning_rate
        # self.use_gpu = params.gpu
        # self.vocab_size = params.vocab_size
        # self.labels_size = params.labels_size

        self.params = params

        # Define the encoder
        self.encoder = encoder.Encoder(
            label_size=params.labels_size,
            hidden_size=params.hidden_size,
            vocab_size=params.vocab_size,
            dropout=params.dropout
        )

        # Define the optimizer, loss function
        self.optimizer = optim.Adam(
            params=self.encoder.parameters(),
            lr=params.learning_rate
        )
        self.loss_function = nn.CrossEntropyLoss()

    def run_train(self, data_loader: DataLoader, data_dir, model_dir):
        """
        Train the model and evaluate every epoch.

        Args:
            data_loader: (DataLoader) store, process the aspects of data
            data_dir: (str) directory containing the data
            model_dir: (string) directory containing config, weights and log
        """
        utils.set_logger(os.path.join(model_dir, 'train.log'))

        # Load data, prepare for training process
        data = data_loader.load_data(
            types=['train', 'val'],
            data_dir=data_dir
        )
        train_data = data['train']
        self.train_size = train_data['size']
        logging.info(" - Done!")
        # Start training

        best_val_acc = 0.0

        logging.info("Starting training for {} epoch(s)".format(self.params.num_epochs))
        for epoch in range(self.params.num_epochs):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, self.params.num_epochs))

            """Training process"""
            # Compute number of batches in one epoch (one full pass over the training set)
            num_steps = (self.train_size + 1) // self.params.batch_size
            train_data_iterator = data_loader.data_iterator(
                data=train_data,
                params=self.params,
                shuffle=False
            )
            # Set model to training mode            
            self.encoder.train()

            # Summary for current training loop and a running average object for loss
            summ = []
            loss_avg = utils.RunningAverage()
            # Use tqdm for progress bar
            progress_batches = trange(num_steps)

            for i in progress_batches:
                # Fetch the next training batch
                sentences_batch, labels_batch = next(train_data_iterator)
            
                # Compute model output and loss
                self.encoder.zero_grad()

                output_batch = self.encoder(sentences_batch)
                loss = self.loss_function(output_batch, labels_batch)
            
                # Compute gradients of loss function with all variables
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
            
                # Performs updates using calculated gradients
                self.optimizer.step()

                # Evaluate summaries only once in a while
                if i % self.params.save_summary_steps == 0:
                    # Extract data from torch Variable, move to cpu, convert to numpy arrays
                    output_batch = output_batch.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()
                    # Compute metrics on this batch
                    summary_batch = {
                        metric: metrics[metric](output_batch, labels_batch)
                        for metric in metrics
                    }
                    summary_batch['loss'] = loss.item()
                    summ.append(summary_batch)

                # Update the average loss
                loss_avg.update(loss.item())
                progress_batches.set_postfix(loss='{:05.3f}'.format(loss_avg()))

            # Compute mean of all metrics in summary
            metrics_mean = {
                metric: np.mean([x[metric] for x in summ])
                for metric in summ[0]
            }
            metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                         for k, v in metrics_mean.items())
            logging.info("- Train metrics: " + metrics_string)            


            """Validate process"""
            val_data = data['val']
            self.val_size = val_data['size']

            val_metrics = self.evaluate(
                data=val_data,
                model_dir=model_dir
            )

            val_acc = val_metrics['accuracy']
            is_best = val_acc >= best_val_acc

            # Save weights
            state = {
                'state_dict': self.encoder.state_dict(),
                'optim_dict': self.optimizer.state_dict()
            }
            utils.save_checkpoint(
                state=state,
                is_best=is_best,
                checkpoint=model_dir
            )

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy!")
                best_val_acc = val_acc
                
                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(
                    model_dir, "metrics_val_best_weights.json"
                )
                utils.save_dict_to_json(val_metrics, best_json_path)        

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(
                model_dir, "metrics_val_last_weights.json"
            )
            utils.save_dict_to_json(val_metrics, last_json_path)


    def evaluate(self, data, model_dir):
        """
        Evaluate for model

        Args:
            data_loader: (DataLoader) store, process the aspects of data
            data_dir: (str) directory containing the data
            model_dir: (string) directory containing config, weights and log
        """
        utils.set_logger(os.path.join(model_dir, 'train.log'))

        # Load data, prepare for training process
        test_size = data['size']

        num_steps = (test_size + 1) // self.params.batch_size
        test_data_iterator = DataLoader.data_iterator(
            data=data,
            params=self.params,
            shuffle=False
        )

        # Set model to evaluation mode
        self.encoder.eval()

        # Summary for current eval loop
        summ = []

        # Compute metrics over the dataset
        for _ in range(num_steps):
            # Fetch the next evaluation batch
            sentences_batch, labels_batch = next(test_data_iterator)

            # Compute model output
            output_batch = self.encoder(sentences_batch)
            loss = self.loss_function(output_batch, labels_batch)

            # Extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # Compute all metrics on this batch
            summary_batch = {
                metric: metrics[metric](output_batch, labels_batch)
                for metric in metrics
            }
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # Compute mean of all metrics in summary
        metrics_mean = {
            metric: np.mean([x[metric] for x in summ])
            for metric in summ[0]
        }
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
        logging.info("- Eval metrics : " + metrics_string)
        return metrics_mean

    # def predict_batch(self, sentences):
    #     """ Predict the sentences by batches

    #     Args:
    #         sentences: (nd.array) array contains the text sentences 
    #     """
    #     self.encoder.eval()
    #     label2prob = []
    #     with torch.no_grad():
    #         batches = data_loader.get_batch(
    #             data=sentences,
    #             batch_size=self.batch_size,
    #             shuffle=False
    #         )
            
    #         for batch in tqdm(batches, total=len(batches), desc='+ Evaluate'.ljust(12, ' ')):
    #             sents_tensor = build_features.preprocess_batch(batch, self.vocab, self.vocab_size)
    #             if self.use_gpu:
    #                 sents_tensor = sents_tensor.cuda()
    #             output = self.encoder(sents_tensor)
    #             output = torch.softmax(output, dim=1)

    #             probs, preds = torch.max(output, 1)
    #             for prob, pred in zip(probs, preds):
    #                 lbl = self.index2label[pred.item()]
    #                 label2prob.append(dict([(lbl, prob.item())]))   
    #         return label2prob


    # def save_model(self, model_path, data_save=None):
    #     # to save the pre-trained model for testing
    #     if data_save is not None:
    #         state = data_save
    #     else:
    #         state = {
    #             'state_dict': self.encoder.state_dict(),
    #             'vocab': self.vocab,
    #             'label2index': self.label2index,
    #             'hidden_size': self.hidden_size,
    #         }
    #     with open(model_path, 'wb') as f:
    #         torch.save(state, f, _use_new_zipfile_serialization=False)

    # def load_model(self, model_file):
    #     # load the pre-trained model
    #     with open(model_file, 'rb') as f:
    #         # If we want to use GPU and CUDA is correctly installed
    #         if self.use_gpu and torch.cuda.is_available():
    #             state = torch.load(f)
    #         else:
    #             # Load all tensors onto the CPU
    #             state = torch.load(f, map_location='cpu')
    #     logger.debug(f'model state = {str(state)}')
    #     if not state:
    #         return None
    #     self.vocab = state['vocab']
    #     self.label2index = state['label2index']
    #     self.hidden_size = state['hidden_size']
    #     self.vocab_size = len(self.vocab)
    #     self.label_size = len(self.label2index)
    #     self.answers = state['answers']
    #     self.origins = state['origins']

    #     self.index2label = {v: k for k, v in self.label2index.items()}

    #     self.encoder = encoder.Encoder(label_size=self.label_size, hidden_size=self.hidden_size,
    #                            max_words=self.vocab_size, dropout=self.dropout)
    #     self.encoder.load_state_dict(state['state_dict'])

    #     if self.use_gpu:
    #         self.encoder.cuda()
    #     return True

    # def count_parameters(self):
    #     return sum(p.numel() for p in self.parameters() if p.requires_grad)


