import os

import configparser
import pandas as pd
from sklearn.metrics import classification_report

import data_loader
import model

MODEL_CONFIG = 'config/trainer/model.cfg'
USE_GPU = False
model_config = configparser.ConfigParser()
model_config.read(MODEL_CONFIG)


def do_train(folder, train_file, val_file, model_name):
    model_path = os.path.join(folder, model_name)

    train_data = data_loader.load_data(os.path.join(folder, train_file), sep=',')
    val_data = data_loader.load_data(os.path.join(folder, val_file))

    config = model_config['params']
    base_model = model.ProductClassify(hidden_size=int(config['hidden_size']),
                               batch_size=int(config['batch_size']),
                               dropout=float(config['dropout']), use_gpu=USE_GPU,
                               lr=float(config['lr']),
                               num_epochs=int(config['num_epochs']))
    
    if len(train_data['sentences']) == 0:
        # Train set is empty
        base_model.save_model(model_path, {})
    base_model.run_train(train_data, val_data, model_path)
    base_model.save_model(model_path)    

    


def do_test_batch(folder, test_file, model_name, unknown_intent, use_semhash):
    model_path = os.path.join(folder, model_name)

    model = model_wrapper.load_model(model_path, model_config['params'], USE_GPU, use_semhash)

    df = pd.read_csv(os.path.join(folder, test_file), delimiter='\t')
    df.dropna(inplace=True)
    samples = df['sample'].values.tolist()
    labels = df['label'].values.tolist()
    result = model_wrapper.infer_batch(model, samples)
    predict_label = [list(pre.keys())[0] for pre in result]
    print(classification_report(labels, predict_label))


def do_infer_batch(folder, test_file, model_name, use_semhash):
    model_path = os.path.join(folder, model_name)

    model = model_wrapper.load_model(model_path, model_config['intent'], USE_GPU, use_semhash)

    df = pd.read_csv(os.path.join(folder, test_file), sep=',')
    samples = df['name'].values.tolist()
    result = model_wrapper.infer_batch(model, samples)
    predict_label = [list(pre.keys())[0] for pre in result]
    predict_probs = [list(pre.values())[0] for pre in result]
    df.insert(loc=len(df.columns), column="predicted_label", value=predict_label)
    df.insert(loc=len(df.columns), column="predicted_probs", value=predict_probs)
    out_file = os.path.join(folder, test_file.replace('.csv', '_predicted.csv'))
    df.to_csv(out_file, sep='\t', index=False)


def do_experiment(folder, train_file, val_file, model_name, use_semhash, mode):
    if mode == 'train':
        do_train(folder, train_file, val_file, model_name, use_semhash)

    if mode == 'test-batch':
        do_test_batch(folder, train_file, model_name,  use_semhash)
    if mode == 'infer-batch':
        do_infer_batch(folder, train_file, model_name, use_semhash)


if __name__ == '__main__':

    folder = 'data'

    csv_file = 'demo/_10k_dev.csv'

    use_semhash = False
    training_with_augument = False    
    model_name = 'model/_240k_kv_0911.pt'
    

    unknown_intent = 'UNKNOWN'

    mode = 'train'
    # mode = 'test-batch'
    # mode = 'infer-batch'
    do_experiment(folder=folder,
                  train_file='demo/_30k_train.csv',
                  val_file='demo/_10k_dev.csv',
                  model_name=model_name,
                  use_semhash=use_semhash,
                  mode=mode)