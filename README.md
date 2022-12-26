# Industry Product Classification - Data Department - Citigo

_Authors: Ngo Phuong Nhi, Nguyen Dinh Nghi, Pham Van Hung, To Duc Anh_

Note : all scripts must be run in `pd-industry-classification`.

## Requirements

We recommend using python3 and a virtual env. See instructions [here](https://docs.python.org/3/library/venv.html).

### Linux
```
python3 -m venv .env
source env/bin/activate
(.env) pip install --upgrade pip
(.env) pip install -r requirements.txt

```
### Windows
```
py -m venv .env
.\env\Scripts\activate
(.env) pip install --upgrade pip
(.env) pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given a product name, give industry name respectively ([Document Classification](https://en.wikipedia.org/wiki/Document_classification))

```
>> infer(["Iphone 14 promax"])
['Điện tử - Điện máy']
```

We provide a demo subset of the csv dataset (100 sentences) for testing in `data/demo_data` and we can go to next steps to **Quickstart**

## [optional] Download the csv dataset (~5 min)

1. **Download the dataset** `.csv` file  on [Google Cloud Storage](https://console.cloud.google.com/storage/browser/bi_recommendation_hub_storage/product_discovery/valitdated_kiotviet?authuser=1) and save it under the `data/kiot-viet` directory. Make sure you file data containing columns: `sentences` and `labels`

2. **Build the dataset** Run the following script

```
python build_data.py --data_dir data/data_demo --data_name product.csv

```

It will extract the sentences and labels from the dataset, split it into train/val/test and save it in a convenient format for our model.

3. **Build** vocabularies and parameters for your dataset by running

```
python build_vocab.py --data_dir data/data_demo --min_count_word 1
```

It will write vocabulary files `words.txt` and `labels.txt` containing the words and labels in the dataset. It will also save a `dataset_params.json` with some extra information.

2. **Your first experiment** We created a `base_model` directory for you under the `experiments` directory. It contains a file `params.json` which sets the hyperparameters for the experiment. It looks like

```yaml
model_config:
  hidden_size: 200
  dropout: 0.3

training_config:
  checkpoint_dir: 'experiments/'
  learning_rate: 0.001
  num_epochs: 10
  train_batch_size: 128
  eval_batch_size: 128
  gpu: -1
  logging_steps: 100
  eval_steps: 1000
  save_steps: 3000
  save_best: True
```

For every new experiment, you will need to create a new directory under `experiments` with a `params.json` file.

3. **Train** your experiment. Simply run

```
python train.py --data_dir data/data_demo --config_path data/config.yaml
```

It will instantiate a model and train it on the training set following the hyperparameters specified in `config.yaml`. It will also evaluate some metrics on the development set.

## Resources

- [PyTorch documentation](http://pytorch.org/docs/1.2.0/)
- [Tutorials](http://pytorch.org/tutorials/)
- [PyTorch warm-up](https://github.com/jcjohnson/pytorch-examples)
