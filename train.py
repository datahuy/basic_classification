from model.model import ClassfierModel
from model.trainer import ClassifierTrainer
from data.classifier_dataset import ClassifierDataset
from torch.utils.data import DataLoader
import argparse
import os
import yaml
import torch

def collate_fn(data):
    text, label = zip(*data)
    
    tokens = list(map(lambda x: [vocab_dict[w] if w in vocab_dict else vocab_dict['UNK'] for w in x.split(' ')], list(text)))
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
    parser.add_argument('--data_dir', default='../classify_data')
    parser.add_argument('--config_path', default='../classify_data/config.yaml')
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

    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        raise ValueError(f"{data_dir} does not exist or is empty.")
        
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        labels = f.read().splitlines()
    label2index = {labels[i]: i for i in range(len(labels))}
    index2label = {i: labels[i] for i in range(len(labels))}

    with open(os.path.join(data_dir, 'words.txt'), 'r') as f:
        vocab = f.read().splitlines()
    vocab_dict = {vocab[i]: i for i in range(len(vocab))}

    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist")

    config = load_config(config_path)
    model_config = config['model_config']
    training_config = config['training_config']
    output_dir = training_config['output_dir']
    # if os.path.exists(output_dir) and os.listdir(output_dir):
    #     raise ValueError(f"{output_dir} exists and is not empty.")
    # else:
    os.makedirs(output_dir, exist_ok=True)

    main()

    # from utils.model_utils import load_model
    # model = load_model('/home/nhinp/Documents/github_repo/pd-industry-classification/output_model/checkpoint_step_10.pt')
    # input = ['quần jeans kids air', 'quần jeans kids air']
    # output = model.predict_batch(input)
    # print(output)