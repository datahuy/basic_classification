import logging
import os
from math import ceil

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
            num_steps = ceil(self.train_size / self.params.batch_size)
            # num_steps = (self.train_size + 1) // self.params.batch_size
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

            val_metrics, val_metrics_string = self.evaluate(data=val_data)
            logging.info("- Eval metrics : " + val_metrics_string)


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


    def evaluate(self, data, infer=False):
        """
        Evaluate for model

        Args:
            data: (dict) contains data which has keys 'sentences', 'labels' and 'size'.
            infer: (bool) whether return labels prediction
        """

        # Load data, prepare for training process
        test_size = data['size']

        num_steps = ceil(test_size / self.params.batch_size)
        test_data_iterator = DataLoader.data_iterator(
            data=data,
            params=self.params,
            shuffle=False
        )


        # Set model to evaluation mode
        self.encoder.eval()

        # Summary for current eval loop
        summ = []

        # Init list containing labels and probability prediction
        pred_labels = []
        pred_probs = []
        # Compute metrics over the dataset
        for _ in trange(num_steps, desc="Inferring"):
            # Fetch the next evaluation batch
            sentences_batch, labels_batch = next(test_data_iterator)
            # Compute model output
            output_batch = self.encoder(sentences_batch)
            loss = self.loss_function(output_batch, labels_batch)
            probs = torch.softmax(output_batch, dim=1)

            # Extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()
            probs = probs.data.cpu().numpy()

            if infer:
                pred_batch = list(np.argmax(output_batch, axis=1))
                pred_labels.extend(pred_batch)
                
                prob_batch = list(np.max(probs, axis=1))
                pred_probs.extend(prob_batch) 

            # Compute all metrics on this batch
            summary_batch = {
                metric: metrics[metric](output_batch, labels_batch)
                for metric in metrics
            }
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        if infer:
            return pred_labels, pred_probs
        
        # Compute mean of all metrics in summary
        metrics_mean = {
            metric: np.mean([x[metric] for x in summ])
            for metric in summ[0]
        }
        metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())

        return metrics_mean, metrics_string