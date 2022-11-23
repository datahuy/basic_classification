from torch.utils.data import Dataset
import os
import numpy as np

class ClassifierDataset(Dataset):
    def __init__(self, data_dir, split, vocab, label2index):
        super(ClassifierDataset, self).__init__()
        with open(os.path.join(data_dir, split, 'sentences.txt'), 'r') as f:
            print(os.path.join(data_dir, split, 'sentences.txt'))
            self.data = f.read().splitlines()
        with open(os.path.join(data_dir, split, 'labels.txt'), 'r') as f:
            self.labels = f.read().splitlines()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.unk_ind = self.vocab['UNK']
        self.label2index = label2index

    def __len__(self):
        return len(self.data)

    
    # def preprocess(self, text):
    #     text = [self.vocab[token] if token in self.vocab else self.unk_ind for token in text.split(' ')]
    #     embedding_text = [0] * self.vocab_size
    #     for token in text:
    #         embedding_text[token] = 1

    #     return np.array(embedding_text, dtype=np.float32)

    def __getitem__(self, idx):
        embedding_text = self.data[idx]
        label = self.label2index[self.labels[idx]]

        return embedding_text, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    with open('../../data/data_demo/words.txt', 'r') as f:
        vocab = f.read().splitlines()

    vocab_dict = {}
    for i, token in enumerate(vocab):
        vocab_dict[token] = i

    train_dataset = ClassifierDataset('../../data/data_demo', 'train', vocab_dict)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    for i, (feature, label) in enumerate(train_dataloader):
        print(label)