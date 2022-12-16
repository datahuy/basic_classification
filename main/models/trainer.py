import json
import torch
from tqdm import tqdm, trange
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


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

        min_loss = float('inf')
        current_steps = 0
        total_steps = len(self.train_dataloader.dataset)
        self.encoder.train()
        for epoch in range(self.num_epochs):
            running_loss = 0
            i = 0
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for text_embeddings, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    self.encoder.zero_grad()

                    outputs = self.encoder(text_embeddings.to(self.device))
                    loss = self.loss_function(outputs, labels.to(self.device))

                    # Compute gradients of loss function with all variables
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), .5)

                    # Performs updates using calculated gradients
                    self.optimizer.step()

                    running_loss += loss.item() * self.train_batch_size
                    if i and i % self.logging_steps == 0:
                        train_summary = {'train_loss': running_loss / (i * self.train_batch_size),
                                         'steps': f"{current_steps}/{total_steps}"}
                        print('\n', train_summary)
                        running_loss = 0
                    i += 1
                    current_steps += 1

                    if current_steps and current_steps % self.eval_steps == 0:
                        eval_summary, _ = self.evaluate()
                        eval_summary['steps'] = f"{current_steps}/{total_steps}"
                        print(eval_summary)
                        if eval_summary['eval_loss'] < min_loss:
                            min_loss = eval_summary['eval_loss']
                            print('Best model so far, eval loss = ', min_loss)
                            self.save_model(os.path.join(self.output_dir, f"best.pt"))

                    if current_steps and current_steps % self.save_steps == 0:
                        self.save_model(os.path.join(self.output_dir, f"checkpoint_step_{current_steps}.pt"))

    def evaluate(self):
        eval_summary = {}
        self.encoder.eval()

        # Compute metrics over the dataset
        running_loss = 0
        all_labels = []
        all_predictions = []
        with tqdm(self.eval_dataloader, unit="batch") as tepoch:
            for text_embeddings, labels in tepoch:
                tepoch.set_description(f"Eval ")
                all_labels.extend(labels.tolist())
                outputs = self.encoder(text_embeddings.to(self.device))
                loss = self.loss_function(outputs, labels.to(self.device))
                running_loss += loss.item() * self.eval_batch_size
                probs = torch.softmax(outputs, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_predictions.extend(preds.tolist())
        eval_summary['eval_loss'] = running_loss / len(self.eval_dataloader.dataset)
        eval_summary['accuracy'] = accuracy_score(all_labels, all_predictions)
        eval_summary['precision'] = precision_score(all_labels, all_predictions, average='macro')
        eval_summary['recall'] = recall_score(all_labels, all_predictions, average='macro')
        eval_summary['f1'] = f1_score(all_labels, all_predictions, average='macro')

        return eval_summary, all_predictions

    def save_model(self, model_path):
        state = {
            'state_dict': self.encoder.state_dict(),
            'vocab_dict': self.model.vocab_dict,
            'index2label': self.model.index2label,
            'hidden_size': self.encoder.hidden_size,
            'dropout': self.encoder.dropout_prob
        }
        with open(model_path, 'wb') as f:
            torch.save(state, f, _use_new_zipfile_serialization=False)
