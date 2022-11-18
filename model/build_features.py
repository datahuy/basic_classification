import numpy as np
import torch
from torch.autograd import Variable



from preprocessing import clean_text


def hash2index(sentences, labels):
    """Create vocab, index for words by vocab

    Args:
        sentences: (string) text senteces
        lables: (string) labels respectively
    """
    data = list()
    vocab = dict()
    label2index = dict()
    vocab['#@UNK@#'] = 0
    for sentence, label in zip(sentences, labels):
        sentence_idx = []
        for hash in sentence.split(" "):
            if hash not in vocab:
                vocab[hash] = len(vocab)
            sentence_idx.append(vocab[hash])

        if label not in label2index:
            label_idx = len(label2index)
            label2index[label] = label_idx

        data.append((sentence_idx, label2index[label]))

    return data, vocab, label2index


def index2matrix(sentence_indices, vocab_size):
    """Convert sentence's index to vectors that have same dimension.

    Args:
        sentence_indices: (nd.array) array contains index of each word
        in a sentence
        vocab_size: (int) number of words in vocab   
    """
    data_size = len(sentence_indices)
    mat = torch.zeros(data_size, vocab_size)

    for idx,sentence_idx in enumerate(sentence_indices):
        index = torch.LongTensor(sentence_idx)
        mat[idx].index_add_(0, index, torch.ones(index.size()))

    mat_var = Variable(mat)

    return mat_var


def prepare_sequences(batch_data, vocab_size=10000):
    labels = []
    sample_indices = []

    for (sample, label) in batch_data:
        sample_indices.append(sample)
        labels.append(label)
    labels = Variable(torch.LongTensor(np.asarray(labels)))
    seq_tensor = index2matrix(sample_indices, vocab_size)

    return seq_tensor, labels


def preprocess_batch(sentences, word2index, vocab_size=10000):
    sentence_ids = []
    for sentence in sentences:
        # if sentence == sentence:
        #     sentence = sentence.replace('_', ' ')
        sentence_idx = []
        sentence = clean_text(sentence)
        for word in sentence.strip().split(' '):
            if word in word2index:
                sentence_idx.append(word2index[word])
            else:
                sentence_idx.append(word2index['#@UNK@#'])
        sentence_ids.append(sentence_idx)

    sent_tensor = index2matrix(sentence_ids, vocab_size)

    return sent_tensor