from tqdm import trange
import logging

import pandas as pd


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    Args:
        log_path: (string) where to log
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def load_csv(csv_file, col_sentences, col_labels, sep):
    """Load csv file, convert data to dictionary

    Args:
        csv_file: path to csv data
        col_sentences: column containing sentences
        col_labels: column containing labels
        sep: delimiter between 2 columns
    """

    df = pd.read_csv(csv_file, sep=sep)
    data = {
        "sentences": list(df[col_sentences].values),
        "labels": list(df[col_labels].values)
    }
    return data


if __name__ == "__main__":

    for x in trange(int(1e7), ncols=100,  colour="green"):
        pass