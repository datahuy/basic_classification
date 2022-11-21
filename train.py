"""Train the model"""

import argparse
import logging
import os
from this import d

import torch

import utils
from model.product_classify import ProductClassify
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/kiot-viet',
                    help='Diretory containing the dataset')
parser.add_argument('--model_dir', default='experiments/base_model',
                    help='Diretory containing params.json')
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


if __name__ == '__main__':

    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path
    ), "No json configuration file found {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(100)
    if params.cuda:
        torch.cuda.manual_seed(100)
    
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # Initialize data_loader
    data_loader = DataLoader(args.data_dir, params)

    # Define the model and optimizer
    model = ProductClassify(params=params)
    
    # Train the model
    model.run_train(
        data_loader=data_loader,
        data_dir=args.data_dir,
        model_dir=args.model_dir
    )