# Product Classification - Data Department - Citigo

_Authors: Le Duc Bao, Ngo Phuong Nhi, Dang Huu Tien, Pham Van Hung, To Duc Anh_
The classification model is originally created by Nguyen Dinh Nghi

## Description
Code for classifying products into categories of industry level, level `n` of each industry.

## How to install and run the project
```
pip install -r requirements.txt
bash run.sh
```
```
curl --location --request POST 'https://datadepartment.citigo.net/pd-industry-classification/industry_cls/' \
--header 'Content-Type: application/json' \
--data-raw '{"product_name": ["sữa đặc", "sữa chua vinamilk không đường", "sữa rửa mặt nivea"], "threshold_KP": 0.5, "threshold_FMCG": 0.5}'
```

## How to train the classification model
1. Dataset has to be in `.csv` file with two columns, one for product name (`sentences`) and one for label (`labels`)

2. **Build the dataset** Run the following script
```
python build_data.py --data_dir DATA_DIR --data_name FILE.CSV

```

It will extract the sentences and labels from the dataset, split it into train/val/test set.

3. **Build vocabularies** by running
```
python build_vocab.py --data_dir data/data_demo --min_count_word 1
```

It will write vocabulary files `words.txt` and `labels.txt` containing the words and labels in the dataset. It will also save a `dataset_params.json` with some extra information.

4. Edit model and training config in `data/config.yaml`

5. **Train** your experiment. Simply run
```
python train.py --data_dir DATA_DIR --config_path data/config.yaml --category CATEGORY
```
