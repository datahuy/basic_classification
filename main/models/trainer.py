import logging
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import utils


class ClassifierTrainer():
    def __init__(self, 
                model,
                train_dataloader,
                eval_dataloader,
                training_config):
        self.__dict__.update(training_config)
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.encoder = model.encoder
        if self.gpu != -1 and torch.cuda.is_available():
            self.encoder.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def train(self):
        self.optimizer = optim.Adam(
            params=self.encoder.parameters(),
            lr=self.learning_rate
        )
        self.loss_function = nn.CrossEntropyLoss()

        utils.set_logger(log_path=os.path.join(self.checkpoint_dir, "train.log"))
        logging.info("Started training {} epoch(s)".format(self.num_epochs))

        best_val_acc = 0.0
        self.encoder.train()
        for epoch in range(self.num_epochs):
            logging.info("\nEpoch {}/{}".format(epoch + 1, self.num_epochs))
            current_batch = 0
            train_loss = 0
            train_acc = 0
            with tqdm(self.train_dataloader, unit="batch", colour="green") as tepoch:
                for text_embeddings, labels in tepoch:

                    tepoch.set_description("+ Training".ljust(10))
                    self.encoder.zero_grad()

                    outputs = self.encoder(text_embeddings.to(self.device))
                    loss = self.loss_function(outputs, labels.to(self.device))
                
                    # Compute gradients of loss function with all variables
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)
                
                    # Performs updates using calculated gradients
                    self.optimizer.step()

                    current_batch += 1

                    train_loss += loss.item()
                    loss_avg = train_loss / current_batch
                    
                    probs = torch.softmax(outputs, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    train_acc += accuracy_score(labels, preds)
                    train_acc_avg = train_acc / current_batch

                    train_summary = {
                        "loss": loss_avg,
                        "accuracy": train_acc_avg
                    }
                    
                    tepoch.set_postfix(
                        loss=train_summary["loss"],
                        acc=train_summary["accuracy"]
                    )

                    """Evaluate, save by steps"""

                    # if current_steps and current_steps % self.eval_steps == 0:
                    #     eval_summary, _ = self.evaluate()
                    #     eval_summary['steps'] = f"{current_steps}/{total_steps}"
                    #     print(eval_summary)
                    #     if eval_summary['eval_loss'] < min_loss:
                    #         min_loss = eval_summary['eval_loss']
                    #         print('Best model so far, eval loss = ', min_loss)
                    #         self.save_model(os.path.join(self.output_dir, f"best.pt"))
                    
                    # if current_steps and current_steps % self.save_steps == 0:
                    #     self.save_model(os.path.join(self.output_dir, f"checkpoint_step_{current_steps}.pt"))
            
            str_summ = "- Train metrics:"
            for k, v in train_summary.items():
                str_summ += "{}: {:05.3f}".format(k, v).rjust(20)
            logging.info(str_summ)
            eval_summary, _ = self.evaluate()
            str_summ = "- Eval metrics :"
            for k, v in eval_summary.items():
                str_summ += "{}: {:05.3f}".format(k, v).rjust(20)
            logging.info(str_summ)
            if eval_summary['accuracy'] > best_val_acc:
                
                logging.info("* Found better model *")
                best_val_acc = eval_summary['accuracy']
                self.save_model(
                    checkpoint_dir=self.checkpoint_dir,
                    is_best=True
                )
        
            # if current_steps and current_steps % self.save_steps == 0:
                # self.save_model(os.path.join(self.output_dir, f"checkpoint_step_{current_steps}.pt"))


    def evaluate(self):
        eval_summary = {}
        self.encoder.eval()

        # Compute metrics over the dataset
        running_loss = 0
        all_labels = []
        all_predictions = []
        with tqdm(self.eval_dataloader, unit="batch", colour="cyan") as tepoch:
            for text_embeddings, labels in tepoch:
                tepoch.set_description("+ Eval".ljust(10))
                all_labels.extend(labels.tolist())
                outputs = self.encoder(text_embeddings.to(self.device))
                loss = self.loss_function(outputs, labels.to(self.device))
                running_loss += loss.item() * self.eval_batch_size
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_predictions.extend(preds.tolist())
        eval_summary['loss'] = running_loss / len(self.eval_dataloader.dataset)
        eval_summary['accuracy'] = accuracy_score(all_labels, all_predictions)
        eval_summary['precision'] = precision_score(all_labels, all_predictions, average='macro')
        eval_summary['recall'] = recall_score(all_labels, all_predictions, average='macro')
        eval_summary['f1'] = f1_score(all_labels, all_predictions, average='macro')

        return eval_summary, all_predictions



    def save_model(self, checkpoint_dir, is_best=False):
        state = {
            'state_dict': self.encoder.state_dict(),
            'vocab_dict': self.model.vocab_dict,
            'index2label': self.model.index2label,
            'hidden_size': self.encoder.hidden_size,
            'dropout': self.encoder.dropout_prob
        }

        file_path = os.path.join(checkpoint_dir, "last.pt")
        with open(file_path, 'wb') as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)
        
        if is_best:
            shutil.copyfile(file_path, os.path.join(checkpoint_dir, 'best.pt'))
