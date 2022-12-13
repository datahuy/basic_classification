import logging
import sys
from typing import Union

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from main.classifier.FMCGl1_classifier import FMCGl1Classifier
import time
from typing import List
from main.rule.rule_mapping import rule_predict_batch
from main.rule.rule_mapping import mapping
from main.rule.rule_mapping import read_json

keywords = read_json("main/rule/data-bin/keyword_lv1.json")
model = FMCGl1Classifier('output_model/l1/best.pt', batch_size=128)


def merge_output(model_output: List, rule_output: List):
    indexes = [idx for idx, j in enumerate(rule_output) if j != "unk"]
    for (index, r_out) in zip(indexes, rule_output):
        model_output[index] = r_out
    return model_output


def pipeline(input: List, keywords, batch_size):
    rule_output = rule_predict_batch(input, batch_size, keywords)
    model_output = model.predict(input, batch_size)
    merged_output = merge_output(model_output, rule_output)
    return merged_output


if __name__ == '__main__':
    test = ["sữa ensure gold 700ml", "sữa chua", "sữa bột", "sữa các loại", "sữa tươi"]
    df = pd.read_csv("data/raw/l1/data_nn_train.csv", sep="\t")
    out = pipeline(test, keywords, batch_size=128)
    print(out)

