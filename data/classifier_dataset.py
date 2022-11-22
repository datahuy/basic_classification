from torch.utils.data import Dataset
import os

class ClassifierDataset(Dataset):
    def __init__(self, data_dir, split, vocab):
        super(ClassifierDataset, self).__init__()
        with open(os.path.join(data_dir, split, 'data.txt'), 'r') as f:
            self.data = f.read.splitlines()
        with open(os.path.join(data_dir, split, 'labels.txt'), 'r') as f:
            self.label = f.read.spitlines()
        self.vocab = vocab
        self.vocab_size = len(vocab)


    def __len__(self):
        return len(self.data)

    
    def preprocess(self, text):
        text = [self.vocab[token] if token in self.vocab else self.unk_ind for token in text.split(' ')]
        embedding_text = [0] * self.vocab_size
        for token in text:
            embedding_text[token] = 1

        return embedding_text

    def __get_item__(self, idx):
        embedding_text = self.preprocess(self.data[idx])
        label = self.labels[idx]

        return embedding_text, label

