import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing import clean_text

def get_batch(data, batch_size, shuffle=True):
    if shuffle:
        random.shuffle(data)
    start_idx = 0
    batches = []
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        batch = data[start_idx:start_idx + batch_size]
        start_idx = start_idx + batch_size
        batches.append(batch)

    if start_idx < len(data):
        batch = data[start_idx:len(data)]
        batches.append(batch)

    return batches


def load_csv(csv_file, col_sentences='name', col_labels='label', sep=','):
    """Load and preprocess the data"""
    
    df = pd.read_csv(csv_file, sep=sep)
    print('- Done!')
    sentences = df[col_sentences].values
    sentences = np.asarray([clean_text(sent) for sent in tqdm(sentences, desc='Preprocessing data')])
    labels = df[col_labels].values
    data = {
        "sentences" : sentences,
        "labels" : labels
    }
    
    return data
