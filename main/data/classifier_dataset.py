from torch.utils.data import Dataset
import os


class ClassifierDataset(Dataset):
    def __init__(self, data_dir, split, label2index):
        super(ClassifierDataset, self).__init__()
        with open(os.path.join(data_dir, split, 'sentences.txt'), 'r') as f:
            self.data = f.read().splitlines()
        with open(os.path.join(data_dir, split, 'labels.txt'), 'r') as f:
            self.labels = f.read().splitlines()
        self.label2index = label2index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding_text = self.data[idx]
        label = self.label2index[self.labels[idx]]

        return embedding_text, label
