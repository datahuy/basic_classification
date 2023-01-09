import logging
from main.models.model import ClassfierModel
from main.models.trainer import ClassifierTrainer
from main.data.classifier_dataset import ClassifierDataset
from torch.utils.data import DataLoader
import argparse
import os
import yaml
import torch

from main.utils.logger import set_logger

def collate_fn(data):
    text, label = zip(*data)

    tokens = list(
        map(lambda x: [vocab_dict[w] if w in vocab_dict else vocab_dict['UNK'] for w in x.split(' ')], list(text)))
    text_embeddings = torch.zeros(len(text), len(vocab_dict))
    for idx, item in enumerate(tokens):
        item = torch.LongTensor(item)
        text_embeddings[idx].index_add_(0, item, torch.ones(item.size()))

    return text_embeddings, torch.LongTensor(label)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Train a product classifier')
    parser.add_argument('--data_dir', default='data/')
    parser.add_argument('--config_path', default='data/config.yaml')
    parser.add_argument('--category', default='l1')
    args = parser.parse_args()

    return args


def main():
    train_dataset = ClassifierDataset(data_dir,
                                      split='train',
                                      label2index=label2index)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=training_config['train_batch_size'],
                                  shuffle=True,
                                  collate_fn=collate_fn)

    eval_dataset = ClassifierDataset(data_dir,
                                     split='test',
                                     label2index=label2index)
    eval_dataloader = DataLoader(eval_dataset,
                                 batch_size=training_config['eval_batch_size'],
                                 shuffle=True,
                                 collate_fn=collate_fn)

    model = ClassfierModel(vocab_dict, index2label, model_config)
    trainer = ClassifierTrainer(model, train_dataloader, eval_dataloader, training_config)
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    data_dir = args.data_dir
    config_path = args.config_path
    category = args.category
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise ValueError(f"{data_dir} does not exist or is empty.")
    
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist")

    # Load config parameteers
    config = load_config(config_path)
    model_config = config['model_config']
    training_config = config['training_config']
    checkpoint_dir = training_config['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set the random seed for reproducible experiments
    torch.manual_seed(100)

    # Set the logger
    log_path = os.path.join(checkpoint_dir, "train.log")
    set_logger(log_path=log_path)

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    if int(training_config["gpu"]) >= 0:
        logging.info("Training on GPU")
        torch.cuda.manual_seed(100)
    else:
        logging.info("Trainging on CPU")
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        labels = f.read().splitlines()
    label2index = {labels[i]: i for i in range(len(labels))}
    print(label2index)
    index2label = {i: labels[i] for i in range(len(labels))}
    print(index2label)
    with open(os.path.join(data_dir, 'words.txt'), 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
    vocab_dict = {vocab[i]: i for i in range(len(vocab))}

    
    # if os.path.exists(output_dir) and os.listdir(output_dir):
    #     raise ValueError(f"{output_dir} exists and is not empty.")
    # else:

    main()

    os.makedirs(output_dir, exist_ok=True)
    main()
